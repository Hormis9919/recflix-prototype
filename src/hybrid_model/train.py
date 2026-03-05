import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import os
import pandas as pd

from src.hybrid_model.hybrid_model      import HybridRecommender
from src.hybrid_model.movietext_dataset import MovieTextDataset, build_static_movie_texts
from src.hybrid_model.vocab             import Vocabulary


def build_vocab_from_reviews(data_dir: Path) -> Vocabulary:
    df = pd.read_csv(
        data_dir / "rotten_tomatoes_movie_reviews.csv"
    ).dropna(subset=["reviewText"])
    tokenized = [str(t).split() for t in df["reviewText"]]
    vocab = Vocabulary(min_freq=2)
    vocab.build(tokenized)
    return vocab


def train():
    torch.backends.cudnn.benchmark = True
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}  |  AMP: {use_amp}")

    DATA_DIR  = Path("datasets/datasets/rotten_tomatoes")
    MODEL_DIR = Path("models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    print("Building vocabulary...")
    vocab = build_vocab_from_reviews(DATA_DIR)

    # ------------------------------------------------------------------
    # Dataset + 80 / 10 / 10 split
    # ------------------------------------------------------------------
    print("Loading dataset...")
    full_dataset = MovieTextDataset(DATA_DIR, vocab, max_len=100)
    n            = len(full_dataset)

    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    n_test  = n - n_train - n_val   # absorbs any rounding remainder

    print(f"Total: {n:,}  ->  "
          f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # seed=42 makes the split reproducible across every run
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Persist all three index lists so evaluate.py uses the exact same split
    torch.save(torch.tensor(train_set.indices), MODEL_DIR / "train_indices.pt")
    torch.save(torch.tensor(val_set.indices),   MODEL_DIR / "val_indices.pt")
    torch.save(torch.tensor(test_set.indices),  MODEL_DIR / "test_indices_hybrid.pt")
    print("Split indices saved to models/")

    print("Building static movie texts for negative sampling...")
    static_texts = build_static_movie_texts(full_dataset, device=device)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    loader_kw = dict(
        num_workers     = 2,
        pin_memory      = use_amp,
        prefetch_factor = 2 if use_amp else None,
    )
    train_loader = DataLoader(train_set, batch_size=512,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=1024,
                              shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = HybridRecommender(
        num_users  = len(full_dataset.critic2idx),
        num_movies = len(full_dataset.movie2idx),
        vocab_size = len(vocab),
        embed_dim  = 64,
        pad_idx    = vocab.token2idx[vocab.PAD_TOKEN],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # ------------------------------------------------------------------
    # Training loop — 5 epochs, validate after each
    # ------------------------------------------------------------------
    print("Training hybrid model (5 epochs)...")
    for epoch in range(5):

        # ---- train ---------------------------------------------------
        model.train()
        train_loss = 0.0

        for ci, mi, rt, sc in train_loader:
            ci = ci.to(device, non_blocking=use_amp)
            mi = mi.to(device, non_blocking=use_amp)
            rt = rt.to(device, non_blocking=use_amp)
            sc = sc.to(device, non_blocking=use_amp)

            # --- 1:3 Negative Sampling ---
            NUM_NEGATIVES = 3
            batch_size = ci.size(0)
            
            # Generate 3x the number of negative movies
            neg_mi = torch.randint(0, len(full_dataset.movie2idx), (batch_size * NUM_NEGATIVES,), device=device)
            neg_rt = static_texts[neg_mi]
            neg_sc = torch.zeros(batch_size * NUM_NEGATIVES, dtype=sc.dtype, device=device)
            
            # Repeat the critic indices 3 times so they match the negative movies
            ci_neg = ci.repeat(NUM_NEGATIVES)
            
            # Combine 1 part positive + 3 parts negative
            ci_all = torch.cat([ci, ci_neg], dim=0)
            mi_all = torch.cat([mi, neg_mi], dim=0)
            rt_all = torch.cat([rt, neg_rt], dim=0)
            sc_all = torch.cat([sc, neg_sc], dim=0)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = model(ci_all, mi_all, rt_all)
                    loss  = criterion(preds, sc_all)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(ci_all, mi_all, rt_all)
                loss  = criterion(preds, sc_all)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        # ---- validate ------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for ci, mi, rt, sc in val_loader:
                ci = ci.to(device, non_blocking=True)
                mi = mi.to(device, non_blocking=True)
                rt = rt.to(device, non_blocking=True)
                sc = sc.to(device, non_blocking=True)

                # Validation with negative sampling
                batch_size = ci.size(0)
                neg_mi = torch.randint(0, len(full_dataset.movie2idx), (batch_size,), device=device)
                neg_rt = static_texts[neg_mi]
                neg_sc = torch.zeros(batch_size, dtype=sc.dtype, device=device)
                
                ci_all = torch.cat([ci, ci], dim=0)
                mi_all = torch.cat([mi, neg_mi], dim=0)
                rt_all = torch.cat([rt, neg_rt], dim=0)
                sc_all = torch.cat([sc, neg_sc], dim=0)
                
                if use_amp:
                    with torch.amp.autocast(device.type):
                        preds = model(ci_all, mi_all, rt_all)
                else:
                    preds = model(ci_all, mi_all, rt_all)
                val_loss += criterion(preds, sc_all).item()

        print(f"Epoch {epoch+1:>2}/5  |  "
              f"Train Loss: {train_loss / len(train_loader):.4f}  |  "
              f"Val Loss:   {val_loss   / len(val_loader):.4f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    print("Saving model...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_users":  len(full_dataset.critic2idx),
        "num_movies": len(full_dataset.movie2idx),
        "vocab_size": len(vocab),
        "embed_dim":  64,
        "pad_idx":    vocab.token2idx[vocab.PAD_TOKEN],
    }, MODEL_DIR / "hybrid_model.pt")

    vocab.save(MODEL_DIR / "hybrid_vocab.pt")
    print("Done. Model and split indices saved to models/")


if __name__ == "__main__":
    train()