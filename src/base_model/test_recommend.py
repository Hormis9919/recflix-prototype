import torch
import pandas as pd
from pathlib import Path

from src.base_model.dataset import MovieLensDataset,load_movielens_ratings
from src.base_model.model import CollaborativeFilteringModel
from src.base_model.recommend import recommend_topn
from src.common_files.utils import get_device

def load_movie_titles(data_dir: Path)->dict:
    movies_path = data_dir/"movies.dat" 
    movies_df = pd.read_csv(movies_path,sep="::",engine="python",names=["movie_id","title","genres"],encoding="latin-1")
    return dict(zip(movies_df["movie_id"],movies_df["title"]))

def main():
    device = get_device()

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "datasets" / "ml-1m"
    MODEL_PATH = ROOT_DIR / "models" / "cf_model.pt"
    movie_id_to_title = load_movie_titles(DATA_DIR)
    ratings_df =  load_movielens_ratings(DATA_DIR)
    dataset = MovieLensDataset(ratings_df)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = CollaborativeFilteringModel(
        num_users=checkpoint["num_users"],
        num_movies=checkpoint["num_movies"],
        embed_dim=checkpoint["embed_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded trained model")
    model.eval()

    print("Recommendations for user 1")
    rec_ids = recommend_topn(model, dataset,user_id=1,n=10,device=device)
    print("Top 10 recommended movie IDs:")
    for rank, movie_id in enumerate(rec_ids,start=1):
        title = movie_id_to_title.get(movie_id,"Unknown Title")
        print(rank,title, sep="\t")

if __name__=="__main__":
    main()    