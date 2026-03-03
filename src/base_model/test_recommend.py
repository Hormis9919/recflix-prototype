import torch
import pandas as pd
from pathlib import Path

from src.base_model.dataset import RTRatingsDataset, load_rt_reviews
from src.base_model.model import CollaborativeFilteringModel
from src.base_model.recommend import recommend_topn
from src.common_files.utils import get_device

def load_movie_titles(data_dir: Path) -> dict:
    movies_path = data_dir / "rotten_tomatoes_movies.csv" 
    # Standard CSV parsing for RT dataset
    movies_df = pd.read_csv(movies_path)
    return dict(zip(movies_df["id"], movies_df["title"]))

def main():
    device = get_device()

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "datasets" / "rotten_tomatoes"
    MODEL_PATH = ROOT_DIR / "models" / "cf_model_rt.pt"
    
    movie_id_to_title = load_movie_titles(DATA_DIR)
    ratings_df = load_rt_reviews(DATA_DIR)
    dataset = RTRatingsDataset(ratings_df)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = CollaborativeFilteringModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        embed_dim=checkpoint["embed_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded trained model")

    # Pass an actual Critic Name from the RT dataset instead of '1'
    target_critic = "Roger Ebert" # Example critic
    print(f"Recommendations for critic: {target_critic}")
    
    try:
        rec_ids = recommend_topn(model, dataset, critic_name=target_critic, n=10, device=device)
        print("Top 10 recommended movie titles:")
        for rank, movie_id in enumerate(rec_ids, start=1):
            title = movie_id_to_title.get(movie_id, f"Unknown Title ({movie_id})")
            print(f"{rank}\t{title}")
    except ValueError as e:
        print(e)

if __name__=="__main__":
    main()