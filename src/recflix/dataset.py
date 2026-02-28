import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset
import re
import numpy as np
from pathlib import Path

# NEW: Import the negative sampler
from src.recflix_alpha.negative_sampler import generate_negative_samples

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
    def __init__(self, explicit_path, vocab, implicit_path=None, max_len=50): # OPTIMIZED
        print("Module 1: Initializing Database Pipeline...")
        self.max_len = max_len
        self.pad_idx = vocab.token2idx[vocab.PAD_TOKEN]
        
        # 1. Load Explicit Data
        print(" -> Loading Explicit Reviews...")
        exp_df = pd.read_csv(explicit_path)
        exp_df["parsed_score"] = exp_df["originalScore"].apply(parse_score)
        exp_df = exp_df.dropna(subset=["parsed_score", "reviewText", "criticName", "id"])
        
        # Standardize columns for merging
        exp_df = exp_df.rename(columns={"criticName": "user_id", "id": "movie_id", "parsed_score": "label"})
        exp_df["data_type"] = "explicit"
        
        # 2. Load Implicit Data (If available)
        if implicit_path and Path(implicit_path).exists():
            print(" -> Loading Implicit Feedback...")
            imp_df = pd.read_csv(implicit_path)
            imp_df["label"] = 1.0 # Positive interaction
            imp_df["reviewText"] = "" # Implicit data has no text
            imp_df["data_type"] = "implicit"
            
            # Combine the datasets
            self.df = pd.concat([exp_df[["user_id", "movie_id", "reviewText", "label", "data_type"]], 
                                 imp_df[["user_id", "movie_id", "reviewText", "label", "data_type"]]], 
                                ignore_index=True)
        else:
            print(" -> No implicit data found. Proceeding with explicit only.")
            self.df = exp_df[["user_id", "movie_id", "reviewText", "label", "data_type"]].reset_index(drop=True)

        # --- OPTIMIZED: INJECT NEGATIVE SAMPLES (Ratio 2) ---
        all_movies_list = self.df["movie_id"].unique().tolist()
        self.df = generate_negative_samples(self.df, all_movies_list, num_negatives=2)
        # -----------------------------------------

        # 3. UNIFIER: Build Global ID Mappings
        print(" -> Unifying Entity Embeddings...")
        self.critic2idx = {critic: idx for idx, critic in enumerate(self.df["user_id"].unique())}
        self.movie2idx = {movie: idx for idx, movie in enumerate(self.df["movie_id"].unique())}

        # 4. Text Processing Pipeline
        print(" -> Processing Text Branch...")
        encoded_texts = []
        self.movie_to_proxy_text = {} 

        for idx, row in self.df.iterrows():
            if row["data_type"] == "explicit":
                tokens = vocab.encode(row["reviewText"])[:max_len]
                if len(tokens) < max_len:
                    tokens += [self.pad_idx] * (max_len - len(tokens))
                
                m_idx = self.movie2idx[row["movie_id"]]
                if m_idx not in self.movie_to_proxy_text:
                    self.movie_to_proxy_text[m_idx] = tokens
            else:
                tokens = [] 
            encoded_texts.append(tokens)

        # 5. Resolve Implicit Text Defaults
        for i, row in self.df.iterrows():
            if row["data_type"] in ["implicit", "implicit_negative"]:
                m_idx = self.movie2idx[row["movie_id"]]
                encoded_texts[i] = self.movie_to_proxy_text.get(m_idx, [self.pad_idx] * max_len)

        # Finalize Tensors
        self.review_tensors = torch.tensor(encoded_texts, dtype=torch.long)
        self.critic_indices = torch.tensor([self.critic2idx[c] for c in self.df["user_id"]], dtype=torch.long)
        self.movie_indices = torch.tensor([self.movie2idx[m] for m in self.df["movie_id"]], dtype=torch.long)
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.float32)

        print(f"Pipeline Ready. Total Samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.critic_indices[idx], 
            self.movie_indices[idx], 
            self.review_tensors[idx], 
            self.labels[idx]
        )
    def save(self, path):
        """Serializes the fully processed dataset to your Drive."""
        print(f" -> Saving processed dataset cache to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Deserializes the dataset instantly from your Drive."""
        print(f" -> Loading dataset from cache: {path}...")
        with open(path, 'rb') as f:
            return pickle.load(f)
