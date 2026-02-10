import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List

from src.prototype.text_utils import tokenize
from src.prototype.vocab import Vocabulary

def pad_sequence(seq: List[int],max_len: int, pad_idx: int) -> List[int]:
    if len(seq)>=max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len-len(seq))

class MovieTextDataset(Dataset):
    def __init__(self, data_dir: Path, vocab: Vocabulary, max_len: int=20,):
        self.vocab = vocab
        self.max_len = max_len

        movies_path = data_dir / "movies.dat"
        movies_df = pd.read_csv(movies_path,sep="::",engine="python",names=["movie_id","title","genres"],encoding="latin-1",)
        self.movie_ids = movies_df["movie_id"].tolist()
        self.encoded_texts = []

        for _, row in movies_df.iterrows():
            text = f"{row['title']} {row['genres']}"
            tokens = tokenize(text)
            encoded = vocab.encode(tokens)
            padded = pad_sequence(encoded,max_len=self.max_len,pad_idx=vocab.token2idx(vocab.PAD_TOKEN),)
            self.encoded_texts.append(padded)
        
        self.encoded_texts = torch.tensor(self.encoded_texts,dtype=torch.long)
        
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        return self.movie_ids[idx],self.encoded_texts[idx]