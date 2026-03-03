import torch
from torch.utils.data import DataLoader
from pathlib import Path
import math

from src.hybrid_model.load_model import load_hybrid_model
from src.hybrid_model.movietext_dataset import MovieTextDataset, build_static_movie_texts
from src.hybrid_model.vocab import Vocabulary
from src.hybrid_model.evaluation import hit_ratio_at_k, ndcg_at_k

def evaluate():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    DATA_DIR = Path("datasets/datasets/rotten_tomatoes")

    print("Loading vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.load("models/hybrid_vocab.pt")

    print("Loading dataset...")
    dataset = MovieTextDataset(DATA_DIR, vocab, max_len=100)
    
    # OPTIMIZATION: Fast DataLoader for evaluation
    dataloader = DataLoader(
        dataset, 
        batch_size=1024, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Loading model...")
    model = load_hybrid_model("models/hybrid_model.pt", device=device)
    model.eval()

    print("Calculating RMSE (using specific reviews)...")
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for critic_idx, movie_idx, review_tensor, score in dataloader:
            critic_idx = critic_idx.to(device, non_blocking=True)
            movie_idx = movie_idx.to(device, non_blocking=True)
            review_tensor = review_tensor.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)

            # OPTIMIZATION: AMP Autocast for faster inference
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(critic_idx, movie_idx, review_tensor)
                
            total_mse += torch.sum((preds - score) ** 2).item()
            total_samples += len(score)

    val_rmse = math.sqrt(total_mse / total_samples)

    print("Building static movie profiles for ranking evaluation...")
    static_movie_text_tensor = build_static_movie_texts(dataset, device=device)

    print("Calculating HR@10 and NDCG@10...")
    
    # We can also wrap the ranking metrics in autocast to speed up the massive tensor operations
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        hr = hit_ratio_at_k(model, dataset, static_movie_text_tensor, device, k=10, threshold=0.8)
        ndcg = ndcg_at_k(model, dataset, static_movie_text_tensor, device, k=10, threshold=0.8)

    print("-" * 30)
    print("Rotten Tomatoes Evaluation Results:")
    print(f"RMSE (0-1 scale) : {val_rmse:.4f}")
    print(f"HR@10            : {hr:.4f}")
    print(f"NDCG@10          : {ndcg:.4f}")

if __name__ == "__main__":
    evaluate()
