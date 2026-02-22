import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.base_model.dataset import MovieLensDataset,load_movielens_ratings
from src.hybrid_model.load_model import load_hybrid_model
from src.hybrid_model.movietext_dataset import MovieTextDataset
from src.hybrid_model.vocab import Vocabulary
from src.common_files.text_utils import tokenize

import pandas as pd
import math

def build_vocab_from_movies(data_dir:Path):
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movies_id","title","genres"],
        encoding="latin-1",
    )
    tokenized = [
        tokenize(f"{t} {g}")
        for t,g in zip(movies["title"],movies["genres"])
    ]
    vocab  = Vocabulary(min_freq=2)
    vocab.build(tokenized)
    return vocab

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path("datasets/ml-1m")

    ratings_df = load_movielens_ratings(DATA_DIR)
    ratings_dataset = MovieLensDataset(ratings_df)
    ratings_loader = DataLoader(ratings_dataset)

    vocab = Vocabulary(min_freq=1)
    vocab.load("models/hybrid_vocab.pt")

    movie_text_dataset = MovieTextDataset(DATA_DIR,vocab,max_len=20)
    movie_text_tensor = movie_text_dataset.encoded_texts.to(device)

    model = load_hybrid_model("models/hybrid_model.pt",device=device)

    mae_loss = nn.L1Loss()
    total_mae = 0
    mse_loss = nn.MSELoss()
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for user_idx, movie_idx, rating in ratings_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            movie_text = movie_text_tensor[movie_idx]
            preds = model(user_idx,movie_idx,movie_text)
            loss = mse_loss(preds,rating)
            mae = mae_loss(preds,rating)
            total_mae += mae.item()*len(rating)
            total_mse += loss.item()*len(rating)
            total_samples += len(rating)

    rmse = math.sqrt(total_mse/total_samples)
    mae = total_mae/total_samples
    print(f"Hybrid MAE: {mae:.4f}")
    print(f"Hybrid RMSE: {rmse:.4f}")

if __name__ == "__main__":
    evaluate()