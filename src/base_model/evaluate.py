import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import math

from src.base_model.dataset import MovieLensDataset, load_movielens_ratings
from src.base_model.model import CollaborativeFilteringModel
from src.common_files.utils import get_device


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = CollaborativeFilteringModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        embed_dim=checkpoint["embed_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate():
    device = get_device()
    print(f"Using device: {device}")

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "datasets" / "ml-1m"

    ratings_df = load_movielens_ratings(DATA_DIR)
    dataset = MovieLensDataset(ratings_df)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model_path = ROOT_DIR / "models" / "cf_model.pt"
    model = load_model(model_path, device)

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_mse = 0
    total_mae = 0
    total_samples = 0

    with torch.no_grad():
        for user_idx, movie_idx, rating in loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            preds = model(user_idx, movie_idx)

            total_mse += mse_loss(preds, rating).item()
            total_mae += mae_loss(preds, rating).item()
            total_samples += len(rating)

    rmse = math.sqrt(total_mse / total_samples)
    mae = total_mae / total_samples

    print(f"\nBase Model RMSE: {rmse:.4f}")
    print(f"Base Model MAE: {mae:.4f}")


if __name__ == "__main__":
    evaluate()