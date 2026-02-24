import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from src.recflix_alpha.vocab import Vocabulary
from src.recflix_alpha.dataset import RecFlixAlphaDataset
from src.recflix_alpha.model import RecFlixAlphaModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROOT = Path(__file__).resolve().parents[2]
    REVIEWS_PATH = ROOT / "datasets/rotten_tomatoes_movie_reviews.csv"
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok = True)

    #building vocabulary
    print("Building vocabulary")
    df = pd.read_csv(REVIEWS_PATH)
    texts = df["reviewText"].dropna().tolist()
    vocab = Vocabulary(min_freq=5)
    vocab.build_vocab(texts)

    #create dataset
    dataset = RecFlixAlphaDataset(
        reviews_path=REVIEWS_PATH,
        vocab=vocab,
        max_len=100
    )

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    #initialize model
    embed_dim = 32
    pad_idx = vocab.token2idx[vocab.PAD_TOKEN]

    model = RecFlixAlphaModel(
        num_critics=len(dataset.critic2idx),
        num_movies=len(dataset.movie2idx),
        vocab_size=len(vocab.token2idx),
        embed_dim=embed_dim,
        pad_idx=pad_idx
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #Training loop
    epochs = 3
    print("Test training(3 epochs)")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for critic_idx, movie_idx, review_tensor, score in train_loader:

            critic_idx = critic_idx.to(device)
            movie_idx = movie_idx.to(device)
            review_tensor = review_tensor.to(device)
            score = score.to(device)

            preds = model(critic_idx, movie_idx, review_tensor)
            loss = criterion(preds, score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
    #Save model and vocab
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_critics": len(dataset.critic2idx),
        "num_movies": len(dataset.movie2idx),
        "vocab_size": len(vocab.token2idx),
        "embed_dim": embed_dim,
        "pad_idx": pad_idx,
    }, MODEL_DIR / "recflix_alpha.pt")
    vocab.save(MODEL_DIR / "recflix_alpha_vocab.pt")
    print("Test training complete")

if __name__ == "__main__":
   train()