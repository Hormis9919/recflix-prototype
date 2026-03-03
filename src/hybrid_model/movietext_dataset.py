import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import re

from src.hybrid_model.vocab import Vocabulary

def parse_score(score):
    if pd.isna(score) or not isinstance(score, str):
        return None
    match = re.match(r"(\d+(\.\d+)?)/(\d+(\.\d+)?)", score)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(3))
        if denominator != 0:
            value = numerator / denominator
            if 0 <= value <= 1:
                return value
    return None

class MovieTextDataset(Dataset):
    def __init__(self, data_dir: Path, vocab: Vocabulary, max_len: int = 100):
        self.vocab = vocab
        self.max_len = max_len

        reviews_path = data_dir / "rotten_tomatoes_movie_reviews.csv"
        print(f"Loading reviews from {reviews_path}...")
        df = pd.read_csv(reviews_path)

        # Parse scores and drop invalid rows
        df["parsed_score"] = df["originalScore"].apply(parse_score)
        df = df.dropna(subset=["parsed_score", "reviewText", "criticName", "id"])
        df = df.reset_index(drop=True)

        self.df = df
        
        # Build mappings
        self.critic2idx = {critic: idx for idx, critic in enumerate(df["criticName"].unique())}
        self.movie2idx = {movie: idx for idx, movie in enumerate(df["id"].unique())}

        # Encode texts
        print("Encoding text...")
        encoded_texts = []
        for text in df["reviewText"]:
            tokens = vocab.encode(str(text).split())
            tokens = tokens[:max_len]
            if len(tokens) < max_len:
                tokens += [vocab.token2idx[vocab.PAD_TOKEN]] * (max_len - len(tokens))
            encoded_texts.append(tokens)

        self.review_tensors = torch.tensor(encoded_texts, dtype=torch.long)
        self.critic_indices = torch.tensor([self.critic2idx[c] for c in df["criticName"]], dtype=torch.long)
        self.movie_indices = torch.tensor([self.movie2idx[m] for m in df["id"]], dtype=torch.long)
        self.scores = torch.tensor(df["parsed_score"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.critic_indices[idx],
            self.movie_indices[idx],
            self.review_tensors[idx],
            self.scores[idx]
        )

def build_static_movie_texts(dataset: MovieTextDataset, device="cpu"):
    """
    Extracts the first seen review for each movie to use as its static text 
    representation during recommendation/evaluation ranking.
    """
    movie_indices = list(dataset.movie2idx.values())
    movie_review_map = {}

    for i in range(len(dataset)):
        _, m_idx, review_tensor, _ = dataset[i]
        m_idx_val = m_idx.item()
        if m_idx_val not in movie_review_map:
            movie_review_map[m_idx_val] = review_tensor

    review_tensor_list = [movie_review_map[m] for m in movie_indices]
    return torch.stack(review_tensor_list).to(device)
