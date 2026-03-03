import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset
import re
import numpy as np
from pathlib import Path

from src.recflix.negative_sampler import generate_negative_samples

def parse_score(score):
    if not isinstance(score, str):
        return None
    match = re.match(r"(\d+(\.\d+)?)/(\d+(\.\d+)?)", score)
    if match:
        numerator, denominator = float(match.group(1)), float(match.group(3))
        if denominator == 0: return None
        value = numerator / denominator
        return value if 0 <= value <= 1 else None
    return None

class UnifiedRecFlixDataset(Dataset):
    def __init__(self, explicit_path, vocab, implicit_path=None, max_len=50): 
        print("Module 1: Initializing Database Pipeline...")
        self.max_len = max_len
        self.pad_idx = vocab.token2idx[vocab.PAD_TOKEN]
        
        # 1. Load Explicit Data
        print(" -> Loading Explicit Reviews...")
        exp_df = pd.read_csv(explicit_path)
        exp_df["parsed_score"] = exp_df["originalScore"].apply(parse_score)
        
        # Safely extract isTopCritic for Fairness metrics
        if "isTopCritic" in exp_df.columns:
            exp_df["isTopCritic"] = exp_df["isTopCritic"].fillna(False).astype(bool)
        else:
            exp_df["isTopCritic"] = False
            
        exp_df = exp_df.dropna(subset=["parsed_score", "reviewText", "criticName", "id"])
        
        # Standardize columns for merging
        exp_df = exp_df.rename(columns={"criticName": "user_id", "id": "movie_id", "parsed_score": "label"})
        exp_df["data_type"] = "explicit"
        
        # 2. Load Implicit Data (If available)
        if implicit_path and Path(implicit_path).exists():
            print(" -> Loading Implicit Feedback...")
            imp_df = pd.read_csv(implicit_path)
            imp_df["label"] = 1.0 
            imp_df["reviewText"] = "" 
            imp_df["data_type"] = "implicit"
            self.df = pd.concat([exp_df, imp_df], ignore_index=True)
        else:
            self.df = exp_df.reset_index(drop=True)

        # --- Inject negative samples (Ratio 3) ---
        all_movies_list = self.df["movie_id"].unique().tolist()
        self.df = generate_negative_samples(self.df, all_movies_list, num_negatives=3)
        # -----------------------------------------

        # 3. UNIFIER: Build Global ID Mappings
        print(" -> Unifying Entity Embeddings...")
        self.critic2idx = {critic: idx for idx, critic in enumerate(self.df["user_id"].unique())}
        self.movie2idx = {movie: idx for idx, movie in enumerate(self.df["movie_id"].unique())}

        # --- NEW: Engineering Beyond-Accuracy Features ---
        print(" -> Engineering Evaluation Metadata (Novelty & Fairness)...")
        # Map which critics are "Top Critics"
        critic_top_map = exp_df.groupby("user_id")["isTopCritic"].max().to_dict()
        self.critic_is_top = {idx: critic_top_map.get(critic, False) for critic, idx in self.critic2idx.items()}
        
        # Map Normalized Movie Popularity (0.0 to 1.0 scale)
        pop_counts = exp_df["movie_id"].value_counts().to_dict()
        max_pop = max(pop_counts.values()) if pop_counts else 1
        self.movie_popularity = {self.movie2idx[m]: count/max_pop for m, count in pop_counts.items() if m in self.movie2idx}
        # -------------------------------------------------

        # 4. Text Processing Pipeline (CPU OPTIMIZED)
        print(" -> Processing Text Branch (Vectorized)...")
        def encode_and_pad(text):
            tokens = vocab.encode(str(text))[:self.max_len]
            return tokens + [self.pad_idx] * (self.max_len - len(tokens))

        explicit_mask = self.df["data_type"] == "explicit"
        explicit_encoded = self.df.loc[explicit_mask, "reviewText"].apply(encode_and_pad)
        
        self.df["encoded_tokens"] = None
        self.df.loc[explicit_mask, "encoded_tokens"] = explicit_encoded
        
        proxy_map = self.df[explicit_mask].drop_duplicates(subset=["movie_id"]).set_index("movie_id")["encoded_tokens"].to_dict()
        default_pad = [self.pad_idx] * self.max_len

        encoded_texts = [
            tokens if d_type == "explicit" else proxy_map.get(m_id, default_pad)
            for tokens, d_type, m_id in zip(self.df["encoded_tokens"], self.df["data_type"], self.df["movie_id"])
        ]

        self.df = self.df.drop(columns=["encoded_tokens"])

        # Finalize Tensors
        self.review_tensors = torch.tensor(encoded_texts, dtype=torch.long)
        self.critic_indices = torch.tensor([self.critic2idx[c] for c in self.df["user_id"]], dtype=torch.long)
        self.movie_indices = torch.tensor([self.movie2idx[m] for m in self.df["movie_id"]], dtype=torch.long)
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.float32)

        print(f"Pipeline Ready. Total Samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    # UNCHANGED: Keeps train.py perfectly intact
    def __getitem__(self, idx):
        return (self.critic_indices[idx], self.movie_indices[idx], self.review_tensors[idx], self.labels[idx])
        
    def save(self, path):
        print(f" -> Saving processed dataset cache to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        print(f" -> Loading dataset from cache: {path}...")
        with open(path, 'rb') as f:
            return pickle.load(f)