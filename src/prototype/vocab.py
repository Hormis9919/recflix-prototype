from collections import Counter
from typing import List, Dict

class Vocabulary:
    def __init__(self,min_freq: int = 2) -> None:
        self.min_freq = min_freq

        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"

        self.token2idx: Dict[int, str] = {self.PAD_TOKEN:0,self.UNK_TOKEN:1,}
        self.idx2token: Dict[str,int] = {0:self.PAD_TOKEN,1:self.UNK_TOKEN,}

    def build(self,tokenized_texts:List[List[str]]):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    def encode(self,tokens:List[str]) -> List[int]:
        return [self.token2idx.get(token,self.token2idx[self.UNK_TOKEN]) for token in tokens]
    def __len__(self):
        return len(self.token2idx)