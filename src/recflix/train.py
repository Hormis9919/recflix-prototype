import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

# CUDA specific optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True # Optimizes convolution/transformer paths

from src.recflix_alpha.vocab import Vocabulary
from src.recflix_alpha.dataset import UnifiedRecFlixDataset
from src.recflix_alpha.model import RecFlixAlphaModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32})")

    ROOT = Path(__file__).resolve().parents[2]
    REVIEWS_PATH = ROOT / "datasets/datasets/rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    print("Building vocabulary")
    df = pd.read_csv(REVIEWS_PATH)
    texts = df["reviewText"].dropna().tolist()
    vocab = Vocabulary(min_freq=5)
    vocab.build_vocab(texts)

    dataset = UnifiedRecFlixDataset(
        explicit_path=REVIEWS_PATH, 
        vocab=vocab,
        max_len=50 # OPTIMIZED
    )

    # Dataloader optimized for GPU transfer
    train_loader = DataLoader(
        dataset, 
        batch_size=2048, # OPTIMIZED
        shuffle=True,
        pin_memory=True,  
        num_workers=2,    
        persistent_workers=True 
    )

    embed_dim = 256
    pad_idx = vocab.token2idx[vocab.PAD_TOKEN]

    model = RecFlixAlphaModel(
        num_critics=len(dataset.critic2idx),
        num_movies=len(dataset.movie2idx),
        vocab_size=len(vocab.token2idx),
        embed_dim=embed_dim,
        pad_idx=pad_idx
    ).to(device)

    # OPTIMIZED: Hardware level graph compiler
    model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) 
    
    epochs = 5 
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.003, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )
    
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Optimized Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for critic_idx, movie_idx, review_tensor, score in train_loader:
            
            critic_idx = critic_idx.to(device, non_blocking=True)
            movie_idx = movie_idx.to(device, non_blocking=True)
            review_tensor = review_tensor.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) 

            with torch.amp.autocast('cuda'):
                preds = model(critic_idx, movie_idx, review_tensor)
                loss = criterion(preds, score)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_critics": len(dataset.critic2idx),
        "num_movies": len(dataset.movie2idx),
        "vocab_size": len(vocab.token2idx),
        "embed_dim": embed_dim,
        "pad_idx": pad_idx,
    }, MODEL_DIR / "recflix.pt")
    
    vocab.save(MODEL_DIR / "recflix_vocab.pt")
    print("Training complete and model saved.")
    
    CACHE_DIR = ROOT / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save(CACHE_DIR / "recflix_dataset_cache.pkl")
    print(f"Dataset cache created at: {CACHE_DIR / 'recflix_dataset_cache.pkl'}")

if __name__ == "__main__":
   train()
