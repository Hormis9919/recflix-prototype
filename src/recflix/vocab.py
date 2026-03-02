import re
from collections import Counter
import torch

class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        
        # 1. OPTIMIZATION: Pre-compile the regex pattern once
        self.clean_pattern = re.compile(r"[^a-z0-9 ]")
        
        # Cache the UNK index to avoid dictionary lookups during encoding
        self.unk_idx = 1
    
    def tokenize(self, text):
        # Module 3: Preprocessing Sub-pipeline
        text = str(text).lower()
        # Use the pre-compiled regex engine
        text = self.clean_pattern.sub(" ", text)
        return text.split()
    
    def build_vocab(self, texts):
        counter = Counter()
        
        for text in texts:
            counter.update(self.tokenize(text))
            
        idx = 2 
        
        for token, freq in counter.items():
            if freq >= self.min_freq:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
                
        print(f"Vocabulary size : {len(self.token2idx)}")
        
    def encode(self, text):
        # Module 3: Cleaned and Tokenized sequences
        tokens = self.tokenize(text)
        
        # 2. OPTIMIZATION: Local variable aliasing for micro-speedups inside loops
        unk = self.unk_idx
        token_map = self.token2idx
        
        return [token_map.get(token, unk) for token in tokens]
        
    def save(self, path):
        torch.save(self.token2idx, path)
        
    def load(self, path):
        self.token2idx = torch.load(path)
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        # Ensure the cached unk_idx is restored accurately
        self.unk_idx = self.token2idx.get(self.UNK_TOKEN, 1)