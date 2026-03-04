import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import re
from src.hybrid_model.vocab import Vocabulary


def parse_score(score):
    if pd.isna(score) or not isinstance(score, str):
        return None
    match = re.match(r"(\d+(\.\d+)?)/(\d+(\.\d+)?)", score)
    if match:
        n = float(match.group(1))
        d = float(match.group(3))
        if d != 0:
            v = n / d
            if 0 <= v <= 1:
                return v
    return None


class MovieTextDataset(Dataset):
    def __init__(self, data_dir: Path, vocab: Vocabulary, max_len: int = 100):
        self.vocab = vocab
        self.max_len = max_len
        reviews_path = data_dir / "rotten_tomatoes_movie_reviews.csv"
        print(f"Loading reviews from {reviews_path}...")
        df = pd.read_csv(reviews_path)
        df["parsed_score"] = df["originalScore"].apply(parse_score)
        df = df.dropna(subset=["parsed_score", "reviewText", "criticName", "id"])
        df = df.reset_index(drop=True)
        self.df = df
        self.critic2idx = {c: i for i, c in enumerate(df["criticName"].unique())}
        self.movie2idx  = {m: i for i, m in enumerate(df["id"].unique())}
        print("Encoding text...")
        pad_id = vocab.token2idx[vocab.PAD_TOKEN]
        encoded = []
        for text in df["reviewText"]:
            tokens = vocab.encode(str(text).split())[:max_len]
            tokens += [pad_id] * (max_len - len(tokens))
            encoded.append(tokens)
        self.review_tensors = torch.tensor(encoded, dtype=torch.long)
        self.critic_indices = torch.tensor(
            [self.critic2idx[c] for c in df["criticName"]], dtype=torch.long)
        self.movie_indices  = torch.tensor(
            [self.movie2idx[m] for m in df["id"]], dtype=torch.long)
        # evaluate.py reads dataset.scores directly
        self.scores = torch.tensor(df["parsed_score"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.critic_indices[idx], self.movie_indices[idx],
                self.review_tensors[idx], self.scores[idx])


def build_static_movie_texts(dataset: MovieTextDataset, device="cpu") -> torch.Tensor:
    num_movies = len(dataset.movie2idx)
    seen = {}
    for i in range(len(dataset)):
        m = dataset.movie_indices[i].item()
        if m not in seen:
            seen[m] = dataset.review_tensors[i]
            if len(seen) == num_movies:
                break
    return torch.stack([seen[m] for m in range(num_movies)]).to(device)