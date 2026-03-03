import pandas as pd
import torch
import re
from torch.utils.data import Dataset
from pathlib import Path

class RTRatingsDataset(Dataset):
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
        
        # ID mappings: Strings to Integer Indices
        self.critic2idx = {critic: idx for idx, critic in enumerate(ratings_df["criticName"].unique())}
        self.movie2idx = {movie_id: idx for idx, movie_id in enumerate(ratings_df["id"].unique())}
        
        # Map original string IDs to indices
        self.ratings_df["critic_idx"] = self.ratings_df["criticName"].map(self.critic2idx)
        self.ratings_df["movie_idx"] = self.ratings_df["id"].map(self.movie2idx)
        
        # Convert ratings to float tensor
        self.ratings = torch.tensor(self.ratings_df["rating"].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, idx):
        user_idx = torch.tensor(self.ratings_df.iloc[idx]["critic_idx"], dtype=torch.long)
        movie_idx = torch.tensor(self.ratings_df.iloc[idx]["movie_idx"], dtype=torch.long)
        rating = self.ratings[idx]
        return user_idx, movie_idx, rating
    
def _parse_rt_score(row) -> float:
    """Normalizes messy RT originalScores to a 1.0 - 5.0 scale."""
    state = str(row.get('reviewState', '')).lower()
    score = str(row.get('originalScore', '')).strip().upper()
    
    # Baseline fallback based on Fresh/Rotten status
    fallback = 4.0 if state == 'fresh' else 2.0
    
    if score == 'NAN' or not score:
        return fallback
        
    # Handle Fractions (e.g., "3/4", "7/10")
    if '/' in score:
        try:
            num_str, den_str = score.split('/')[:2]
            # Clean non-numeric characters (e.g., extracting 3 from "3 stars")
            num = float(re.sub(r'[^\d.]', '', num_str))
            den = float(re.sub(r'[^\d.]', '', den_str))
            if den > 0 and num <= den:
                return (num / den) * 5.0
        except ValueError:
            pass
            
    # Handle Letter Grades
    grades = {'A+': 5.0, 'A': 5.0, 'A-': 4.5, 'B+': 4.0, 'B': 3.5, 'B-': 3.0, 
              'C+': 2.5, 'C': 2.0, 'C-': 1.5, 'D+': 1.2, 'D': 1.0, 'F': 0.5}
    if score in grades:
        return grades[score]
        
    return fallback

def load_rt_reviews(data_dir: Path) -> pd.DataFrame:
    reviews_path = data_dir / "rotten_tomatoes_movie_reviews.csv"
    df = pd.read_csv(reviews_path)
    
    # Drop rows without a critic or a movie ID
    df = df.dropna(subset=['criticName', 'id'])
    
    # Parse scores into a clean 'rating' column
    df['rating'] = df.apply(_parse_rt_score, axis=1)
    
    return df[['criticName', 'id', 'rating']]