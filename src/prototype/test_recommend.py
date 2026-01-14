import torch
import pandas as pd
from pathlib import Path

from src.prototype.dataset import MovieLensDataset,load_movielens_ratings
from src.prototype.model import CollaborativeFilteringModel
from src.prototype.recommend import recommend_topn
from src.prototype.utils import get_device

def load_movie_titles(data_dir: Path)->dict:
    movies_path = data_dir/"movies.dat" 
    movies_df = pd.read_csv(movies_path,sep="::",engine="python",names=["movie_id","title","genres"],encoding="latin-1")
    return dict(zip(movies_df["movie_id"],movies_df["title"]))

def main():
    device = get_device()

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "datasets" / "ml-1m"
    movie_id_to_title = load_movie_titles(DATA_DIR)
    ratings_df =  load_movielens_ratings(DATA_DIR)
    dataset = MovieLensDataset(ratings_df)
    model = CollaborativeFilteringModel(num_users=len(dataset.user2idx),num_movies=len(dataset.movie2idx),embed_dim=32).to(device)

    print("Recommendations for user 1")
    rec_ids = recommend_topn(model, dataset,user_id=1,n=10,device=device)
    print("Top 10 recommended movie IDs:")
    for rank, movie_id in enumerate(rec_ids,start=1):
        title = movie_id_to_title.get(movie_id,"Unknown Title")
        print(rank,title, sep="\t")

if __name__=="__main__":
    main()    