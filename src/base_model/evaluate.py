import torch
import math
from torch.utils.data import DataLoader
from pathlib import Path

from src.base_model.dataset import MovieLensDataset, load_movielens_ratings
from src.base_model.model import CollaborativeFilteringModel
from src.base_model.evaluation import hit_ratio_at_k, ndcg_at_k
from src.base_model import config


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "datasets/ml-1m"
    MODEL_PATH = ROOT / "models/cf_model.pt"

    # Load dataset
    ratings_df = load_movielens_ratings(DATA_DIR)
    dataset = MovieLensDataset(ratings_df)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # Recreate model architecture
    model = CollaborativeFilteringModel(
        num_users=len(dataset.user2idx),
        num_movies=len(dataset.movie2idx),
        embed_dim=config.USER_EMBED_DIM,
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Evaluating...")

    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for user_idx, movie_idx, rating in loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            preds = model(user_idx, movie_idx)

            total_mse += torch.sum((preds - rating) ** 2).item()
            total_samples += len(rating)

    val_rmse = math.sqrt(total_mse / total_samples)

    # Ranking metrics
    hr = hit_ratio_at_k(model, dataset, device, k=10)
    ndcg = ndcg_at_k(model, dataset, device, k=10)

    print(f"\nRMSE: {val_rmse:.4f}")
    print(f"HR@10: {hr:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")


if __name__ == "__main__":
    evaluate()