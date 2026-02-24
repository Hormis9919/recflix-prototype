import torch
import torch.nn as nn
from src.recflix_alpha.review_encoder import ReviewEncoder


class RecFlixAlphaModel(nn.Module):
    def __init__(self, num_critics, num_movies, vocab_size, embed_dim, pad_idx):
        super().__init__()

        # Embeddings
        self.critic_embedding = nn.Embedding(num_critics, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)

        # Review encoder
        self.review_encoder = ReviewEncoder(vocab_size, embed_dim, pad_idx)

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize embeddings (important for stability)
        nn.init.normal_(self.critic_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, critic_idx, movie_idx, review_tensor):

        # Get embeddings
        critic_vec = self.critic_embedding(critic_idx)
        movie_vec = self.movie_embedding(movie_idx)
        review_vec = self.review_encoder(review_tensor)

        # Concatenate
        x = torch.cat([critic_vec, movie_vec, review_vec], dim=1)

        # MLP
        rating_pred = self.mlp(x)

        # Sigmoid to bound between 0 and 1
        rating_pred = torch.sigmoid(rating_pred)

        return rating_pred.squeeze(1)