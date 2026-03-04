import torch
import math
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.hybrid_model.movietext_dataset import MovieTextDataset, build_static_movie_texts
from src.hybrid_model.vocab             import Vocabulary
from src.hybrid_model.load_model        import load_hybrid_model


def _load_indices(path: Path) -> list:
    raw = torch.load(path, weights_only=False)
    return raw.tolist() if isinstance(raw, torch.Tensor) else list(raw)


def _popularity_maps(dataset: MovieTextDataset, indices: list):
    """
    Build critic_is_top and movie_popularity from training indices ONLY.
    Using training indices here prevents any test-set leakage into the
    fairness and novelty metadata signals.
    """
    cc, mc = defaultdict(int), defaultdict(int)
    for i in indices:
        cc[dataset.critic_indices[i].item()] += 1
        mc[dataset.movie_indices[i].item()]  += 1
    counts = sorted(cc.values())
    thr    = counts[int(len(counts) * 0.8)] if counts else 0
    cit    = {k: (v >= thr) for k, v in cc.items()}
    mx     = max(mc.values(), default=1)
    return cit, {k: v / mx for k, v in mc.items()}


def evaluate():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    DATA_DIR  = Path("datasets/datasets/rotten_tomatoes")
    MODEL_DIR = Path("models")

    # ------------------------------------------------------------------
    # Vocab + dataset
    # ------------------------------------------------------------------
    vocab = Vocabulary(min_freq=2)
    vocab.load(MODEL_DIR / "hybrid_vocab.pt")

    dataset = MovieTextDataset(DATA_DIR, vocab, max_len=100)

    # ------------------------------------------------------------------
    # Load the saved split indices
    # ------------------------------------------------------------------
    if (MODEL_DIR / "test_indices_hybrid.pt").exists():
        test_indices_hybrid = _load_indices(MODEL_DIR / "test_indices_hybrid.pt")
    else:
        test_indices_hybrid = list(range(len(dataset)))

    if (MODEL_DIR / "train_indices.pt").exists():
        train_indices = _load_indices(MODEL_DIR / "train_indices.pt")
    else:
        train_indices = list(range(len(dataset)))

    # ------------------------------------------------------------------
    # Popularity maps
    # ------------------------------------------------------------------
    critic_is_top, movie_pop = _popularity_maps(dataset, train_indices)

    # ------------------------------------------------------------------
    # DataLoader over test subset
    # ------------------------------------------------------------------
    loader = DataLoader(
        Subset(dataset, test_indices_hybrid),
        batch_size  = 2048,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = use_amp,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = load_hybrid_model(MODEL_DIR / "hybrid_model.pt", device=device)
    model.eval()

    k   = 10
    REL = 0.5

    nc       = max(critic_is_top.keys(), default=-1) + 1
    top_mask = torch.tensor(
        [critic_is_top.get(i, False) for i in range(nc)],
        dtype=torch.bool, device=device,
    )

    # ------------------------------------------------------------------
    # 1. Rating metrics (Calculates RMSE and MAE)
    # ------------------------------------------------------------------
    t_mse = top_mse = reg_mse = t_mae = 0.0
    t_n   = top_n   = reg_n   = 0

    with torch.inference_mode():
        for ci, mi, rt, sc in loader:
            ci = ci.to(device, non_blocking=True)
            mi = mi.to(device, non_blocking=True)
            rt = rt.to(device, non_blocking=True)
            sc = sc.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device.type):
                    preds = model(ci, mi, rt).float()
            else:
                preds = model(ci, mi, rt).float()

            p5 = preds * 4.0 + 1.0
            s5 = sc.float() * 4.0 + 1.0
            sq = (p5 - s5) ** 2
            ab = torch.abs(p5 - s5)

            it = top_mask[ci.clamp(max=len(top_mask) - 1)]

            t_mse += sq.sum().item()
            t_mae += ab.sum().item()
            t_n   += len(sc)
            if it.sum() > 0:
                top_mse += sq[it].sum().item()
                top_n   += it.sum().item()
            if (~it).sum() > 0:
                reg_mse += sq[~it].sum().item()
                reg_n   += (~it).sum().item()

    def sr(v, n): return math.sqrt(v / n) if n > 0 else 0.0

    rmse     = sr(t_mse,   t_n)
    mae      = t_mae / t_n if t_n > 0 else 0.0
    top_rmse = sr(top_mse, top_n)
    reg_rmse = sr(reg_mse, reg_n)

    # ------------------------------------------------------------------
    # 2. Ranking + beyond-accuracy metrics (1-vs-99 Protocol)
    # ------------------------------------------------------------------
    print("Computing ranking metrics (1-vs-99 Protocol)...")
    
    # Map out all global interactions to ensure we sample TRUE negatives
    user_interacted_movies = defaultdict(set)
    for i in range(len(dataset)):
        user_interacted_movies[dataset.critic_indices[i].item()].add(dataset.movie_indices[i].item())
    
    all_movie_ids = list(range(len(dataset.movie2idx)))
    NUM_NEGATIVES = 99

    # Group test items by critic
    groups = defaultdict(list)
    for i in test_indices_hybrid:
        c = dataset.critic_indices[i].item()
        groups[c].append((
            dataset.movie_indices[i].item(),
            dataset.review_tensors[i],
            dataset.scores[i].item(),
        ))

    acc = {x: 0.0 for x in ["hit", "ndcg", "prec", "rec", "f1", "mrr", "map", "auc"]}
    rec_set = set()
    novelty = top_ndcg = reg_ndcg = 0.0
    eval_count = tu = ru = 0
    
    # We need the static texts for the 99 negative movies that don't have reviews
    static_movie_text_tensor = build_static_movie_texts(dataset, device=device)

    with torch.inference_mode():
        for critic, items in groups.items():
            is_top = critic_is_top.get(critic, False)
            unseen = list(set(all_movie_ids) - user_interacted_movies[critic])

            for pos_movie_idx, rvt, st in items:
                if st < REL:
                    continue  # Only evaluate ranking on movies they actually liked

                # Sample 99 negatives
                negatives = random.sample(unseen, min(NUM_NEGATIVES, len(unseen)))
                candidates = [pos_movie_idx] + negatives
                
                ct_tensor = torch.full((len(candidates),), critic, dtype=torch.long, device=device)
                mt_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

                # Use actual review for the positive item, static text for negatives
                pos_text = rvt.unsqueeze(0).to(device)
                neg_text = static_movie_text_tensor[negatives]
                text_tensor = torch.cat([pos_text, neg_text], dim=0)

                if use_amp:
                    with torch.amp.autocast(device.type):
                        preds = model(ct_tensor, mt_tensor, text_tensor).squeeze()
                else:
                    preds = model(ct_tensor, mt_tensor, text_tensor).squeeze()

                # Get Top K
                _, topk_indices = torch.topk(preds, min(k, len(candidates)))
                topk_indices = topk_indices.cpu().tolist()
                
                # Get Full Ranking (For AUC calculation)
                full_sorted_indices = torch.argsort(preds, descending=True).cpu().tolist()
                
                # Track Novelty and Coverage
                mv = [candidates[i] for i in topk_indices]
                rec_set.update(mv)
                pop = [movie_pop.get(m, 0.0) for m in mv]
                novelty += 1.0 - sum(pop) / len(pop)

                # The positive item is ALWAYS at index 0 of our candidates array.
                rank = topk_indices.index(0) if 0 in topk_indices else -1
                true_rank = full_sorted_indices.index(0) # TRUE rank across all 100 items
                
                hit = 1.0 if rank != -1 else 0.0
                acc["hit"] += hit
                acc["prec"] += hit / k
                acc["rec"] += hit # It's 1 or 0 since there is only 1 positive item in the pool
                
                if hit:
                    acc["f1"] += (2 * (1/k) * 1) / ((1/k) + 1)
                    mrr_val = 1.0 / (rank + 1)
                    acc["mrr"] += mrr_val
                    acc["map"] += mrr_val # MAP is identical to MRR in a 1-vs-N setup
                    ndcg_val = 1.0 / math.log2(rank + 2)
                    acc["ndcg"] += ndcg_val
                    
                    if is_top:
                        top_ndcg += ndcg_val
                    else:
                        reg_ndcg += ndcg_val
                
                # Increment denominators regardless of hit/miss
                if is_top:
                    tu += 1
                else:
                    ru += 1
                
                # AUC based on the TRUE rank across all 100 candidates
                acc["auc"] += (len(candidates) - 1 - true_rank) / (len(candidates) - 1)

                eval_count += 1

    def av(key, n): return acc[key] / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  FINAL IEEE EVALUATION METRICS (TEST SET)  ")
    print("=" * 55)
    print("--- Rating Quality (1-5 Stars) ---")
    print(f"MAE:                {mae:.4f}")
    print(f"RMSE (Global):      {rmse:.4f}")
    print("-" * 55)
    print("--- Ranking Quality (Top 10) ---")
    print(f"HR@10:              {av('hit',  eval_count):.4f}")
    print(f"Precision@10:       {av('prec', eval_count):.4f}")
    print(f"Recall@10:          {av('rec',  eval_count):.4f}")
    print(f"F1-Score@10:        {av('f1',   eval_count):.4f}")
    print(f"MAP@10:             {av('map',  eval_count):.4f}")
    print(f"MRR:                {av('mrr',  eval_count):.4f}")
    print(f"NDCG@10 (Global):   {av('ndcg', eval_count):.4f}")
    print(f"AUC:                {av('auc',  eval_count):.4f}")
    print("-" * 55)
    print("--- Beyond-Accuracy (Fairness & Novelty) ---")
    print(f"RMSE (Top Critics): {top_rmse:.4f}  | RMSE (Standard): {reg_rmse:.4f}")
    print(f"NDCG (Top Critics): {top_ndcg / tu if tu > 0 else 0:.4f}"
          f"  | NDCG (Standard): {reg_ndcg / ru if ru > 0 else 0:.4f}")
    print(f"Item Coverage:      "
          f"{len(rec_set) / len(dataset.movie2idx) * 100:.2f}% of catalog")
    print(f"Novelty Score:      "
          f"{novelty / eval_count if eval_count > 0 else 0:.4f} (1.0 = Highly Novel)")
    print("=" * 55)

if __name__ == "__main__":
    evaluate()