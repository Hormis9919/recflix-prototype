import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import math
from collections import defaultdict

from src.recflix_alpha.dataset import RecFlixAlphaDataset
from src.recflix_alpha.vocab import Vocabulary
from src.recflix_alpha.load_model import load_recflix_alpha_model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROOT = Path(__file__).resolve().parents[2]
    REVIEWS_PATH = ROOT / "datasets/rotten_tomatoes_movie_reviews.csv"
    MODEL_DIR = ROOT / "models"

    #load vocab
    vocab = Vocabulary(min_freq=1)
    vocab.load(MODEL_DIR / "recflix_alpha_vocab.pt")

    #Load dataset
    dataset = RecFlixAlphaDataset(
        reviews_path=REVIEWS_PATH,
        vocab=vocab,
        max_len=100
    )

    loader = DataLoader(dataset, batch_size=512)
    
    #Load model
    model = load_recflix_alpha_model(
        MODEL_DIR / "recflix_alpha.pt",
        device=device
    )
    #RMSE Calculation
    total_mse = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for critic_idx, movie_idx, review_tensor, score in loader:

            critic_idx = critic_idx.to(device)
            movie_idx = movie_idx.to(device)
            review_tensor = review_tensor.to(device)
            score = score.to(device)

            preds = model(critic_idx, movie_idx, review_tensor)

            total_mse += torch.sum((preds - score) ** 2).item()
            total_samples += len(score)

            all_preds.append(preds.cpu())
            all_targets.append(score.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    rmse = math.sqrt(total_mse / total_samples)

    #Ranking Metrics
    k = 10
    critic_groups = defaultdict(list)

    for i in range(len(dataset)):
        critic_idx, movie_idx, review_tensor, score = dataset[i]
        critic_groups[int(critic_idx)].append((movie_idx, review_tensor, score))

    hit_count = 0
    ndcg_total = 0
    critic_count = 0

    with torch.no_grad():
        for critic, items in critic_groups.items():

            if len(items) < k:
                continue

            critic_tensor = torch.tensor([critic] * len(items)).to(device)
            movie_tensor = torch.stack([x[0] for x in items]).to(device)
            review_tensor = torch.stack([x[1] for x in items]).to(device)
            true_scores = torch.tensor([x[2] for x in items])

            preds = model(critic_tensor, movie_tensor, review_tensor)
            preds = preds.cpu()

            # Sort predictions
            topk_indices = torch.topk(preds, k).indices

            # Relevant = true score > 0.7
            relevant = (true_scores > 0.7)

            # HR@K
            if torch.any(relevant[topk_indices]):
                hit_count += 1

            # NDCG@K
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

    # Print Results
    print(f"RMSE: {rmse:.4f}")
    print(f"HR@{k}: {hr:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")


if __name__ == "__main__":
    evaluate()