import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os

from src.base_model.dataset import MovieLensDataset, load_movielens_ratings
from src.hybrid_model.hybrid_model import HybridRecommender
from src.hybrid_model.movietext_dataset import MovieTextDataset
from src.hybrid_model.vocab import Vocabulary
from src.common_files.text_utils import tokenize

import pandas as pd

def build_vocab_from_movies(data_dir: Path):
    movies = pd.read_csv(data_dir / "movies.dat", sep="::",engine="python",names=["movie_id","title","genres"],encoding="latin-1",)
    tokenized = [tokenize(f"{t} {g}")
                 for t,g in zip(movies["title"],movies["genres"])]
    vocab = Vocabulary(min_freq=2)
    vocab.build(tokenized)
    return vocab

def train():
    device = torch.device("cuda" if torch.cuda.is_available()else"cpu")
    print("Using device:",device)
    DATA_DIR = Path("datasets/ml-1m")

    ratings_df = load_movielens_ratings(DATA_DIR)
    ratings_dataset = MovieLensDataset(ratings_df)
    ratings_loader = DataLoader(ratings_dataset, batch_size = 256, shuffle=True)

    vocab = build_vocab_from_movies(DATA_DIR)

    movie_text_dataset = MovieTextDataset(DATA_DIR, vocab, max_len=20)

    movie_text_tensor = movie_text_dataset.encoded_texts.to(device)

    model = HybridRecommender(num_users=len(ratings_dataset.user2idx),num_movies=len(ratings_dataset.movie2idx),vocab_size=len(vocab),embed_dim=64,pad_idx=vocab.token2idx[vocab.PAD_TOKEN],).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.MSELoss()

    print("Training hybrid model...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for user_idx, movie_idx, rating in ratings_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            movie_text = movie_text_tensor[movie_idx]

            optimizer.zero_grad()

            preds = model(user_idx, movie_idx, movie_text)
            loss = criterion(preds,rating)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch{epoch+1} | Loss: {total_loss / len(ratings_loader):.4f}")
    print("Hyrid training complete.")
    os.makedirs("models",exist_ok=True)
    torch.save({
        "model_state_dict":model.state_dict(),
        "num_users": len(ratings_dataset.user2idx),
        "num_movies": len(ratings_dataset.movie2idx),
        "vocab_size": len(vocab),
        "embed_dim": 64,
        "pad_idx": vocab.token2idx[vocab.PAD_TOKEN]
    },"models/hybrid_model.pt")
    vocab.save("models/hybrid_vocab.pt")
    print("Hybrid model saved to datasets folder.")

if __name__ == "__main__":
    train()