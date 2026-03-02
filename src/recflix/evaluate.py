import os
import torch
import math
from collections import defaultdict
from torch.utils.data import DataLoader
from pathlib import Path

# RTX 30 Series (Ampere) specific optimizations for inference
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

    # 2. Optimized Dataset Loading with Cache Check
    CACHE_PATH = ROOT / "cache/recflix_dataset_cache.pkl"

    print("Starting fast dataset boot...")
    if CACHE_PATH.exists():
        # BOOT TIME: ~5 Seconds
        dataset = UnifiedRecFlixDataset.load(CACHE_PATH)
    else:
        # BOOT TIME: ~15 Minutes (Only runs if cache is missing)
        print("No cache found. Processing CSV (First-time only)...")
        dataset = UnifiedRecFlixDataset(
            explicit_path=REVIEWS_PATH,
            vocab=vocab,
            max_len=50 
        )
        # Optional: Save it now so next time is fast
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(CACHE_PATH)

    loader = DataLoader(
        dataset, 
        batch_size=2048, # OPTIMIZED
        pin_memory=True,
        num_workers=2
    )
    
    model = load_recflix_model(
        MODEL_DIR / "recflix.pt", # OPTIMIZED: Load final run
        device=device
    )

    total_mse = 0
    total_samples = 0

    print("Computing RMSE...")
    model.eval()
    
    # OPTIMIZED: Strict inference mode
    with torch.inference_mode():
        for critic_idx, movie_idx, review_tensor, score in loader:
            critic_idx = critic_idx.to(device, non_blocking=True)
            movie_idx = movie_idx.to(device, non_blocking=True)
            review_tensor = review_tensor.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(critic_idx, movie_idx, review_tensor)
                preds = torch.sigmoid(logits) # Apply sigmoid manually for display/metrics
            
            preds = preds.float() * 5.0
            sccore = score.float() * 5.0
            
            total_mse += torch.sum((preds - score) ** 2).item()
            total_samples += len(score)

    rmse = math.sqrt(total_mse / total_samples)

    print("Computing Ranking Metrics (HR@10, NDCG@10)...")
    k = 10
    critic_groups = defaultdict(list)

    for i in range(len(dataset)):
        critic_idx, movie_idx, review_tensor, score = dataset[i]
        critic_groups[int(critic_idx)].append((movie_idx, review_tensor, score))

    hit_count = 0
    ndcg_total = 0
    critic_count = 0

    # OPTIMIZED: Strict inference mode
    with torch.inference_mode():
        for critic, items in critic_groups.items():
            if len(items) < k:
                continue

            critic_tensor = torch.tensor([critic] * len(items)).to(device)
            movie_tensor = torch.stack([x[0] for x in items]).to(device)
            review_tensor = torch.stack([x[1] for x in items]).to(device)
            true_scores = torch.tensor([x[2] for x in items])

            with torch.amp.autocast('cuda'):
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
