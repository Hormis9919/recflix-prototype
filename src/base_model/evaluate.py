import torch
import math
from torch.utils.data import DataLoader
from pathlib import Path

from src.base_model.dataset import RTRatingsDataset, load_rt_reviews
from src.base_model.model import CollaborativeFilteringModel
from src.base_model.evaluation import hit_ratio_at_k, ndcg_at_k
from src.base_model import config

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROOT = Path(__file__).resolve().parents[3]
    # Update to the new dataset and model paths
    DATA_DIR = ROOT / "datasets/datasets" / "rotten_tomatoes"
    MODEL_PATH = ROOT / "models" / "cf_model_rt.pt"

    # Load dataset
    print("Loading ratings for evaluation...")
    ratings_df = load_rt_reviews(DATA_DIR)
    dataset = RTRatingsDataset(ratings_df)
    
    # Using config BATCH_SIZE for consistency
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Recreate model architecture using the new critic/movie counts
    model = CollaborativeFilteringModel(
        num_users=len(dataset.critic2idx),
        num_movies=len(dataset.movie2idx),
        embed_dim=config.USER_EMBED_DIM,
    ).to(device)

    # Load trained weights
    print(f"Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Evaluating...")

    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for critic_idx, movie_idx, rating in loader:
            critic_idx = critic_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            preds = model(critic_idx, movie_idx)

            total_mse += torch.sum((preds - rating) ** 2).item()
            total_samples += len(rating)

    val_rmse = math.sqrt(total_mse / total_samples)

    # Ranking metrics (Threshold 4.0 equals a "B+" or solid "Fresh" rating in our parser)
    hr = hit_ratio_at_k(model, dataset, device, k=10, threshold=4.0)
    ndcg = ndcg_at_k(model, dataset, device, k=10, threshold=4.0)

    print(f"\nRMSE: {val_rmse:.4f}")
    print(f"HR@10: {hr:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    evaluate()