import torch
import torch.nn as nn
class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users:int, num_movies:int, embed_dim:int = 32):
        super().__init__()
        #Embedding layers
        self.user_embedding = nn.Embedding(num_users,embed_dim)
        self.movie_embedding = nn.Embedding(num_movies,embed_dim)
        #Prediction MLP
        self.mlp = nn.Sequential(nn.Linear(embed_dim*2,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.normal_(self.user_embedding.weight,std=0.01)
        nn.init.normal_(self.movie_embedding.weight,std=0.01)
    def forward(self,user_idx,movie_idx):
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)
        #concatenate movie and user embeddings
        x = torch.cat([user_vec,movie_vec],dim=1)
        rating_pred = self.mlp(x)
        return rating_pred.squeeze(1)