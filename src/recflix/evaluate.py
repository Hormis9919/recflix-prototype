import os
import torch
import math
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# Ampere / PyTorch Optimizations
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

    vocab = Vocabulary(min_freq=5)
    vocab.load(MODEL_DIR / "recflix_vocab.pt")

    CACHE_PATH = ROOT / "cache/recflix_dataset_cache.pkl"

    print("Starting fast dataset boot...")
    if CACHE_PATH.exists():
        dataset = UnifiedRecFlixDataset.load(CACHE_PATH)
    else:
        print("No cache found. Processing CSV (First-time only)...")
        dataset = UnifiedRecFlixDataset(
            explicit_path=REVIEWS_PATH,
            vocab=vocab,
            max_len=50 
        )
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(CACHE_PATH)

    # OPTIMIZATION: Load the 10% unseen test indices from training
    test_indices_path = MODEL_DIR / "test_indices.pt"
    if test_indices_path.exists():
        print("Loading 10% Test Split...")
        # weights_only=False added for safety loading legacy tensors
        test_indices = torch.load(test_indices_path, weights_only=False) 
        test_subset = Subset(dataset, test_indices)
    else:
        print("[WARNING] Test indices not found. Evaluating on full dataset.")
        test_subset = dataset
        test_indices = list(range(len(dataset))) # Fallback

    loader = DataLoader(
        test_subset, 
        batch_size=2048,
        pin_memory=True,
        num_workers=2
    )
    
    model = load_recflix_model(
        MODEL_DIR / "recflix.pt",
        device=device
    )

    total_mse = 0
    total_samples = 0

    print("Computing RMSE...")
    model.eval()
    
    with torch.inference_mode():
        for critic_idx, movie_idx, review_tensor, score in loader:
            critic_idx = critic_idx.to(device, non_blocking=True)
            movie_idx = movie_idx.to(device, non_blocking=True)
            review_tensor = review_tensor.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)

            # OPTIMIZATION: device.type enables seamless fallback to your Ryzen CPU
            with torch.amp.autocast(device.type):
                logits = model(critic_idx, movie_idx, review_tensor)
                preds = torch.sigmoid(logits) 
            
            preds = preds.float() * 5.0
            score = score.float() * 5.0
            
            total_mse += torch.sum((preds - score) ** 2).item()
            total_samples += len(score)

    rmse = math.sqrt(total_mse / total_samples)

    print("Computing Ranking Metrics (HR@10, NDCG@10)...")
    k = 10
    
    # OPTIMIZATION: Bypassing __getitem__ completely for ranking
    critic_groups = defaultdict(list)
    
    print("Grouping unseen evaluation data...")
    # Extract native tensors strictly aligned with the test indices
    c_indices = dataset.critic_indices[test_indices].tolist()
    m_indices = dataset.movie_indices[test_indices]
    r_tensors = dataset.review_tensors[test_indices]
    true_scores = dataset.labels[test_indices]

    # Zipped iteration creates zero class-method overhead
    for i, critic in enumerate(c_indices):
        critic_groups[critic].append((m_indices[i], r_tensors[i], true_scores[i]))

    hit_count = 0
    ndcg_total = 0
    critic_count = 0

    with torch.inference_mode():
        for critic, items in critic_groups.items():
            if len(items) < k:
                continue

            # OPTIMIZATION: Direct GPU memory allocation, bypassing Python lists
            critic_tensor = torch.full((len(items),), critic, dtype=torch.long, device=device)
            movie_tensor = torch.stack([x[0] for x in items]).to(device)
            review_tensor = torch.stack([x[1] for x in items]).to(device)
            true_scores = torch.tensor([x[2] for x in items])

            with torch.amp.autocast(device.type):
                preds = model(critic_tensor, movie_tensor, review_tensor)
            
            preds = preds.cpu().float()
            topk_indices = torch.topk(preds, k).indices
            relevant = (true_scores > 0.7)

            if torch.any(relevant[topk_indices]):
                hit_count += 1

            dcg = 0
            idcg = 0
            sorted_true = torch.sort(true_scores, descending=True).values[:k]

            for rank, idx in enumerate(topk_indices):
                if true_scores[idx] > 0.7:
                    dcg += 1 / math.log2(rank + 2)

            for rank, val in enumerate(sorted_true):
                if val > 0.7:
                    idcg += 1 / math.log2(rank + 2)

            if idcg > 0:
                ndcg_total += dcg / idcg

            critic_count += 1

    hr = hit_count / critic_count if critic_count > 0 else 0
    ndcg = ndcg_total / critic_count if critic_count > 0 else 0

    print("-" * 20)
    print(f"RMSE:     {rmse:.4f}")
    print(f"HR@{k}:    {hr:.4f}")
    print(f"NDCG@{k}:  {ndcg:.4f}")

if __name__ == "__main__":
    evaluate()