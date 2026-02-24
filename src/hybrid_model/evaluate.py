import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.base_model.dataset import MovieLensDataset,load_movielens_ratings
from src.hybrid_model.load_model import load_hybrid_model
from src.hybrid_model.movietext_dataset import MovieTextDataset
from src.hybrid_model.vocab import Vocabulary
from src.hybrid_model.evaluation import rmse, hit_ratio_at_k, ndcg_at_k


import pandas as pd
import math

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path("datasets/ml-1m")

    ratings_df = load_movielens_ratings(DATA_DIR)
    ratings_dataset = MovieLensDataset(ratings_df)
    ratings_loader = DataLoader(ratings_dataset, batch_size=1024,shuffle=False)

    # Load vocab (DO NOT rebuild)
    vocab = Vocabulary(min_freq=1)
    vocab.load("models/hybrid_vocab.pt")

    movie_text_dataset = MovieTextDataset(DATA_DIR, vocab, max_len=20)
    movie_text_tensor = movie_text_dataset.encoded_texts.to(device)

    model = load_hybrid_model("models/hybrid_model.pt", device=device)
    model.eval()

    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for user_idx, movie_idx, rating in ratings_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            movie_text = movie_text_tensor[movie_idx]
            preds = model(user_idx, movie_idx, movie_text)

            total_mse += torch.sum((preds - rating) ** 2).item()
            total_samples += len(rating)

        val_rmse = math.sqrt(total_mse / total_samples)

    # Ranking metrics
    hr = hit_ratio_at_k(model, ratings_dataset, movie_text_tensor, device, k=10)
    ndcg = ndcg_at_k(model, ratings_dataset, movie_text_tensor, device, k=10)

    print(f"RMSE: {val_rmse:.4f}")
    print(f"HR@10: {hr:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    evaluate()