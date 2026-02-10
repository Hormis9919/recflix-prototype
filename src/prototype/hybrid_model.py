import torch
import torch.nn as nn
from src.prototype.text_encoder import TextEncoder

class HybridRecommender(nn.Module):
    def __init__(self, num_users: int, num_movies: int, vocab_size: int, embed_dim: int, pad_idx: int, hidden_dim: int=128,):
        super().__init__()
        
        #collborative filtering embeddings
        self.user_embedding = nn.Embedding(num_users,embed_dim)
        self.movie_embedding = nn.Embedding(num_movies,embed_dim)

        #movie metadate textencoder
        self.text_encoder = TextEncoder(vocab_size=vocab_size,embed_dim=embed_dim,pad_idx=pad_idx)

        #prediction network
        self.mlp = nn.Sequential(nn.Linear(embed_dim*3,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1),)

        #initialization
        nn.init.normal_(self.user_embedding.weight,std=0.01)
        nn.init.normal_(self.movie_embedding.weight,std=0.01)
    def forward(self,user_idx: torch.Tensor,movie_idx: torch.Tensor,movie_text: torch.Tensor)-> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)
        text_vec = self.text_encoder(movie_text)
        #concatenate the representations
        x= torch.cat([user_vec,movie_vec,text_vec],dim=1)
        rating_pred = self.mlp(x)
        return rating_pred.squeeze(1)