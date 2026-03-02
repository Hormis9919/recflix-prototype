import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import pandas as pd

# CUDA specific optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True # Optimizes convolution/transformer paths

# OPTIMIZATION: Forced FlashAttention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from src.recflix.vocab import Vocabulary
from src.recflix.dataset import UnifiedRecFlixDataset
from src.recflix.model import RecFlixAlphaModel

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
        max_len=50 
    )

    # 80/10/10 Data Split
    print("Splitting Dataset (80% Train, 10% Val, 10% Test)...")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Random split with fixed seed for reproducibility
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    # Save test indices so evaluate.py can evaluate strictly on unseen data
    torch.save(test_subset.indices, MODEL_DIR / "test_indices.pt")

    # DataLoaders for Train and Validation
    train_loader = DataLoader(
        train_subset, 
        batch_size=2048, 
        shuffle=True,
        pin_memory=True,  
        num_workers=2,    
        persistent_workers=True 
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=2048, 
        shuffle=False,
        pin_memory=True,  
        num_workers=2
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

    model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) 
    
    epochs = 25 
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.003, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )
    
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Optimized Training...")
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        
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
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.inference_mode():
            for critic_idx, movie_idx, review_tensor, score in val_loader:
                critic_idx = critic_idx.to(device, non_blocking=True)
                movie_idx = movie_idx.to(device, non_blocking=True)
                review_tensor = review_tensor.to(device, non_blocking=True)
                score = score.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    preds = model(critic_idx, movie_idx, review_tensor)
                    val_loss = criterion(preds, score)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

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