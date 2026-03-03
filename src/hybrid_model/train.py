import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import pandas as pd

from src.hybrid_model.hybrid_model import HybridRecommender
from src.hybrid_model.movietext_dataset import MovieTextDataset
from src.hybrid_model.vocab import Vocabulary

def build_vocab_from_reviews(data_dir: Path):
    reviews_path = data_dir / "rotten_tomatoes_movie_reviews.csv"
    df = pd.read_csv(reviews_path).dropna(subset=["reviewText"])
    
    tokenized = [str(text).split() for text in df["reviewText"]]
    vocab = Vocabulary(min_freq=2)
    vocab.build(tokenized)
    return vocab

def train():
    # OPTIMIZATION: Enable cuDNN benchmark for faster dynamic graph execution
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Optimized for T4)")
    
    DATA_DIR = Path("datasets/datasets/rotten_tomatoes") 

    print("Building vocabulary...")
    vocab = build_vocab_from_reviews(DATA_DIR)

    print("Loading dataset...")
    dataset = MovieTextDataset(DATA_DIR, vocab, max_len=100)
    
    # OPTIMIZATION: pin_memory and num_workers prevent GPU starvation
    dataloader = DataLoader(
        dataset, 
        batch_size=512, # Increased batch size to saturate T4 VRAM
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )

    model = HybridRecommender(
        num_users=len(dataset.critic2idx),
        num_movies=len(dataset.movie2idx),
        vocab_size=len(vocab),
        embed_dim=64,
        pad_idx=vocab.token2idx[vocab.PAD_TOKEN],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # OPTIMIZATION: Initialize Gradient Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    print("Training hybrid model...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for critic_idx, movie_idx, review_tensor, score in dataloader:
            # OPTIMIZATION: non_blocking=True allows async transfers
            critic_idx = critic_idx.to(device, non_blocking=True)
            movie_idx = movie_idx.to(device, non_blocking=True)
            review_tensor = review_tensor.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)

            # OPTIMIZATION: set_to_none=True is faster and uses less memory
            optimizer.zero_grad(set_to_none=True)

            # OPTIMIZATION: AMP Autocast Context
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(critic_idx, movie_idx, review_tensor)
                loss = criterion(preds, score)

            # Scale gradients and step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")
        
    print("Hybrid training complete.")
    
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_users": len(dataset.critic2idx),
        "num_movies": len(dataset.movie2idx),
        "vocab_size": len(vocab),
        "embed_dim": 64,
        "pad_idx": vocab.token2idx[vocab.PAD_TOKEN]
    }, "models/hybrid_model.pt")
    
    vocab.save("models/hybrid_vocab.pt")
    print("Hybrid model saved.")

if __name__ == "__main__":
    train()
