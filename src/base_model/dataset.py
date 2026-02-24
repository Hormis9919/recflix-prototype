import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MovieLensDataset(Dataset):
    def __init__(self,ratings_df:pd.DataFrame):
        self.ratings_df = ratings_df
        #ID mappings
        self.user2idx = {user_id: idx for idx, user_id in enumerate(ratings_df["user_id"].unique())}
        self.movie2idx = {movie_id: idx for idx, movie_id in enumerate(ratings_df["movie_id"].unique())}
        #map original ids to indices
        self.ratings_df["user_idx"] = ratings_df["user_id"].map(self.user2idx)
        self.ratings_df["movie_idx"] = ratings_df["movie_id"].map(self.movie2idx)
        #convert ratings to float tensor
        self.ratings = torch.tensor(self.ratings_df["rating"].values,dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self,idx):
        user_idx = torch.tensor(self.ratings_df.iloc[idx]["user_idx"],dtype=torch.long)
        movie_idx = torch.tensor(self.ratings_df.iloc[idx]["movie_idx"],dtype=torch.long)
        rating = self.ratings[idx]
        return user_idx, movie_idx,rating
    
def load_movielens_ratings(data_dir: Path) -> pd.DataFrame:
    ratings_path = data_dir / "ratings.dat"
    ratings = pd.read_csv(ratings_path,sep="::",engine="python",names=["user_id","movie_id","rating","timestamp"])
    return ratings