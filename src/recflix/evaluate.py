import os
import torch
import math
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.recflix.dataset import UnifiedRecFlixDataset
from src.recflix.vocab import Vocabulary
from src.recflix.load_model import load_recflix_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device} (TF32: {torch.backends.cuda.matmul.allow_tf32})")

    ROOT = Path(__file__).resolve().parents[2]
    REVIEWS_PATH = ROOT / "datasets/datasets/rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"
    MODEL_DIR = ROOT / "models"
    CACHE_PATH = ROOT / "cache/recflix_dataset_cache.pkl"

    vocab = Vocabulary(min_freq=5)
    vocab.load(MODEL_DIR / "recflix_vocab.pt")

    print("Starting fast dataset boot...")
    if CACHE_PATH.exists():
        dataset = UnifiedRecFlixDataset.load(CACHE_PATH)
    else:
        print("No cache found. Processing CSV (First-time only)...")
        dataset = UnifiedRecFlixDataset(explicit_path=REVIEWS_PATH, vocab=vocab, max_len=50)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(CACHE_PATH)

    test_indices_path = MODEL_DIR / "test_indices.pt"
    if test_indices_path.exists():
        loaded_indices = torch.load(test_indices_path, weights_only=False)
        test_indices = loaded_indices.tolist() if isinstance(loaded_indices, torch.Tensor) else list(loaded_indices)
        test_subset = Subset(dataset, test_indices)
    else:
        test_subset = dataset
        test_indices = list(range(len(dataset)))

    loader = DataLoader(test_subset, batch_size=2048, pin_memory=True, num_workers=2)
    model = load_recflix_model(MODEL_DIR / "recflix.pt", device=device)

    # --- 1. RATING & FAIRNESS METRICS ---
    print("Computing Rating Metrics & Critic Fairness...")
    model.eval()
    
    total_mse, top_mse, reg_mse = 0, 0, 0
    total_samples, top_samples, reg_samples = 0, 0, 0
    total_mae = 0
    
    # Preload the fairness mask to device
    top_critic_tensor = torch.tensor([dataset.critic_is_top.get(i, False) for i in range(len(dataset.critic2idx))], dtype=torch.bool, device=device)

    with torch.inference_mode():
        for critic_idx, movie_idx, review_tensor, score in loader:
            critic_idx = critic_idx.to(device, non_blocking=True)
            movie_idx = movie_idx.to(device, non_blocking=True)
            review_tensor = review_tensor.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)

            with torch.amp.autocast(device.type):
                logits = model(critic_idx, movie_idx, review_tensor)
                preds = torch.sigmoid(logits).float()
            
            valid_mask = score > 0.0
            
            if valid_mask.sum() > 0:
                valid_preds = (preds[valid_mask] * 4.0) + 1.0
                valid_scores = (score[valid_mask].float() * 4.0) + 1.0
                valid_critics = critic_idx[valid_mask]
                
                # Fairness Split
                is_top = top_critic_tensor[valid_critics]
                
                if is_top.sum() > 0:
                    top_mse += torch.sum((valid_preds[is_top] - valid_scores[is_top]) ** 2).item()
                    top_samples += is_top.sum().item()
                    
                if (~is_top).sum() > 0:
                    reg_mse += torch.sum((valid_preds[~is_top] - valid_scores[~is_top]) ** 2).item()
                    reg_samples += (~is_top).sum().item()
                
                total_mse += torch.sum((valid_preds - valid_scores) ** 2).item()
                total_mae += torch.sum(torch.abs(valid_preds - valid_scores)).item()
                total_samples += valid_mask.sum().item()

    rmse = math.sqrt(total_mse / total_samples) if total_samples > 0 else 0
    top_rmse = math.sqrt(top_mse / top_samples) if top_samples > 0 else 0
    reg_rmse = math.sqrt(reg_mse / reg_samples) if reg_samples > 0 else 0
    mae = total_mae / total_samples if total_samples > 0 else 0

    # --- 2. RANKING & NOVELTY METRICS ---
    print("Computing Ranking & Novelty Metrics...")
    k = 10
    critic_groups = defaultdict(list)
    
    c_indices = dataset.critic_indices[test_indices].tolist()
    m_indices = dataset.movie_indices[test_indices]
    r_tensors = dataset.review_tensors[test_indices]
    true_scores = dataset.labels[test_indices]

    for i, critic in enumerate(c_indices):
        critic_groups[critic].append((m_indices[i], r_tensors[i], true_scores[i]))

    metrics = {"hit": 0, "ndcg": 0, "prec": 0, "rec": 0, "f1": 0, "mrr": 0, "map": 0, "auc": 0}
    
    # Beyond-Accuracy Accumulators
    recommended_item_set = set()
    total_novelty = 0
    top_ndcg, reg_ndcg = 0, 0
    
    valid_rank_users, valid_auc_users = 0, 0
    top_users, reg_users = 0, 0
    REL_THRESH = 0.5 

    with torch.inference_mode():
        for critic, items in critic_groups.items():
            if len(items) < k: continue

            critic_tensor = torch.full((len(items),), critic, dtype=torch.long, device=device)
            movie_tensor = torch.stack([x[0] for x in items]).to(device)
            review_tensor = torch.stack([x[1] for x in items]).to(device)
            t_scores = torch.stack([torch.as_tensor(x[2]) for x in items])

            with torch.amp.autocast(device.type):
                logits = model(critic_tensor, movie_tensor, review_tensor)
                preds = torch.sigmoid(logits).cpu().float()
            
            relevance = (t_scores >= REL_THRESH)
            total_rel_user = relevance.sum().item()
            if total_rel_user == 0: continue
                
            valid_rank_users += 1
            is_top_critic = dataset.critic_is_top.get(critic, False)
            
            topk_indices = torch.topk(preds, k).indices
            rel_in_topk = relevance[topk_indices]
            
            # Metadata: Coverage & Novelty
            topk_movies = movie_tensor[topk_indices].cpu().tolist()
            recommended_item_set.update(topk_movies)
            
            pop_scores = [dataset.movie_popularity.get(m, 0.0) for m in topk_movies]
            avg_pop = sum(pop_scores) / len(pop_scores)
            total_novelty += (1.0 - avg_pop) # High score = rare movies
            
            hits_in_k = rel_in_topk.sum().item()

            prec_k = hits_in_k / k
            rec_k = hits_in_k / total_rel_user
            metrics["prec"] += prec_k
            metrics["rec"] += rec_k
            metrics["f1"] += (2 * prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0

            if hits_in_k > 0:
                metrics["hit"] += 1
                first_hit_idx = rel_in_topk.nonzero(as_tuple=True)[0][0].item()
                metrics["mrr"] += 1.0 / (first_hit_idx + 1)
                
            sum_prec = 0
            running_hits = 0
            for rank_idx, is_rel in enumerate(rel_in_topk):
                if is_rel:
                    running_hits += 1
                    sum_prec += running_hits / (rank_idx + 1)
            metrics["map"] += sum_prec / min(k, total_rel_user)

            dcg = sum([relevance[idx].item() / math.log2(rank_idx + 2) for rank_idx, idx in enumerate(topk_indices)])
            idcg = sum([val.item() / math.log2(rank_idx + 2) for rank_idx, val in enumerate(torch.sort(relevance.float(), descending=True).values[:k])])
                
            if idcg > 0:
                user_ndcg = dcg / idcg
                metrics["ndcg"] += user_ndcg
                
                # Metadata: Critic Fairness
                if is_top_critic:
                    top_ndcg += user_ndcg
                    top_users += 1
                else:
                    reg_ndcg += user_ndcg
                    reg_users += 1

            if total_rel_user < len(t_scores):
                pos_preds = preds[relevance]
                neg_preds = preds[~relevance]
                metrics["auc"] += (pos_preds.unsqueeze(1) > neg_preds.unsqueeze(0)).float().mean().item()
                valid_auc_users += 1

    def avg(metric_key, count):
        return metrics[metric_key] / count if count > 0 else 0.0

    print("\n" + "=" * 55)
    print("  FINAL IEEE EVALUATION METRICS (TEST SET)  ")
    print("=" * 55)
    print("--- Rating Quality (1-5 Stars) ---")
    print(f"MAE:                {mae:.4f}")
    print(f"RMSE (Global):      {rmse:.4f}")
    print("-" * 55)
    print(f"--- Ranking Quality (Top {k}) ---")
    print(f"HR@{k}:              {avg('hit', valid_rank_users):.4f}")
    print(f"Precision@{k}:       {avg('prec', valid_rank_users):.4f}")
    print(f"Recall@{k}:          {avg('rec', valid_rank_users):.4f}")
    print(f"F1-Score@{k}:        {avg('f1', valid_rank_users):.4f}")
    print(f"MAP@{k}:             {avg('map', valid_rank_users):.4f}")
    print(f"MRR:                {avg('mrr', valid_rank_users):.4f}")
    print(f"NDCG@{k} (Global):   {avg('ndcg', valid_rank_users):.4f}")
    print(f"AUC:                {avg('auc', valid_auc_users):.4f}")
    print("-" * 55)
    print("--- Beyond-Accuracy Metrics (Fairness & Bias) ---")
    print(f"RMSE (Top Critics): {top_rmse:.4f}  | RMSE (Standard): {reg_rmse:.4f}")
    print(f"NDCG (Top Critics): {top_ndcg/top_users if top_users > 0 else 0:.4f}  | NDCG (Standard): {reg_ndcg/reg_users if reg_users > 0 else 0:.4f}")
    print(f"Item Coverage:      {(len(recommended_item_set) / len(dataset.movie2idx)) * 100:.2f}% of catalog")
    print(f"Novelty Score:      {total_novelty / valid_rank_users if valid_rank_users > 0 else 0:.4f} (1.0 = Highly Novel)")
    print("=" * 55)

if __name__ == "__main__":
    evaluate()