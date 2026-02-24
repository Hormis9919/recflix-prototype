import pandas as pd
import torch
from torch.utils.data import Dataset
import re


def parse_score(score):

    if not isinstance(score, str):
        return None

    match = re.match(r"(\d+(\.\d+)?)/(\d+(\.\d+)?)", score)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(3))

        if denominator == 0:
            return None

        value = numerator / denominator

        # filter normalized scores
        if 0 <= value <= 1:
            return value
        else:
            return None

    return None


class RecFlixAlphaDataset(Dataset):
    def __init__(self, reviews_path, vocab, max_len=200):

        print("Loading Rotten Tomatoes reviews...")
        df = pd.read_csv(reviews_path)

        print("Parsing scores...")
        df["parsed_score"] = df["originalScore"].apply(parse_score)

        # Remove rows without valid score or review text
        df = df.dropna(subset=["parsed_score", "reviewText", "criticName", "id"])

        df = df.reset_index(drop=True)

        print(f"Remaining samples after cleaning: {len(df)}")

        self.df = df
        self.max_len = max_len

        # Build critic and movie mappings
        self.critic2idx = {
            critic: idx for idx, critic in enumerate(df["criticName"].unique())
        }

        self.movie2idx = {
            movie: idx for idx, movie in enumerate(df["id"].unique())
        }

        # Encode review text
        print("Encoding review texts...")
        encoded_texts = []

        for text in df["reviewText"]:
            tokens = vocab.encode(text)
            tokens = tokens[:max_len]

            if len(tokens) < max_len:
                tokens += [vocab.token2idx[vocab.PAD_TOKEN]] * (max_len - len(tokens))

            encoded_texts.append(tokens)

        self.review_tensors = torch.tensor(encoded_texts, dtype=torch.long)

        # Convert critic and movie ids to indices
        self.critic_indices = torch.tensor(
            [self.critic2idx[c] for c in df["criticName"]],
            dtype=torch.long
        )

        self.movie_indices = torch.tensor(
            [self.movie2idx[m] for m in df["id"]],
            dtype=torch.long
        )

        self.scores = torch.tensor(
            df["parsed_score"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.critic_indices[idx],
            self.movie_indices[idx],
            self.review_tensors[idx],
            self.scores[idx]
        )