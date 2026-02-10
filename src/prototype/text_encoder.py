import torch 
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self,vocab_size: int,embed_dim: int,pad_idx: int,):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,padding_idx=pad_idx)
    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_indices)
        mask = (text_indices != self.embedding.padding_idx).unsqueeze(-1)
        embedded = embedded * mask
        summed = embedded.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed/lengths
    