"""
Module: recflix_alpha.model
Purpose: Neural model definition for RecFlix Alpha.
"""

import torch
import torch.nn as nn
import math

# --- Module 2: Collaborative Feature Extractor ---
class CollaborativeFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # MLP Path
        self.mlp_path = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, user_emb, movie_emb):
        # GMF Path (Element-wise multiplication)
        gmf_vec = user_emb * movie_emb
        
        # MLP Path (Concatenation -> Dense)
        mlp_vec = self.mlp_path(torch.cat([user_emb, movie_emb], dim=-1))
        
        # Collaborative Feature Vector
        return torch.cat([gmf_vec, mlp_vec], dim=-1) # Shape: (batch, embed_dim * 2)

# --- Module 4: Lightweight Transformer Encoder ---
class LightweightTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Input Embeddings Merger (Combines User/Movie vectors to condition the text)
        self.context_merger = nn.Linear(embed_dim * 2, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, text_seq, user_emb, movie_emb, pad_idx):
        # 1. Embed text
        text_emb = self.embedding(text_seq)
        
        # 2. Merge user/movie context into a single "CLS" style token
        context_vec = self.context_merger(torch.cat([user_emb, movie_emb], dim=-1))
        context_token = context_vec.unsqueeze(1) # Shape: (batch, 1, embed_dim)
        
        # 3. Prepend context token to sequence
        merged_seq = torch.cat([context_token, text_emb], dim=1)
        
        # 4. Create padding mask for the transformer (ignore PAD tokens)
        pad_mask = (text_seq == pad_idx)
        # Add a "False" at the beginning of mask for our context token
        context_mask = torch.zeros((text_seq.shape[0], 1), dtype=torch.bool, device=text_seq.device)
        full_mask = torch.cat([context_mask, pad_mask], dim=1)
        
        # 5. Pass through Transformer
        encoded_seq = self.transformer(merged_seq, src_key_padding_mask=full_mask)
        
        # 6. Pooling Layer (Extract the context-aware merged token)
        return encoded_seq[:, 0, :] # Shape: (batch, embed_dim)

# --- Module 5: Gated Fusion Mechanism ---
class GatedFusion(nn.Module):
    def __init__(self, collab_dim, content_dim, out_dim):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(collab_dim + content_dim, out_dim),
            nn.Sigmoid()
        )
        self.collab_proj = nn.Linear(collab_dim, out_dim)
        self.content_proj = nn.Linear(content_dim, out_dim)
        
    def forward(self, collab_vec, content_vec):
        # Calculate Gate Weights (g)
        g = self.gate_network(torch.cat([collab_vec, content_vec], dim=-1))
        
        # Gated Summation
        return g * self.collab_proj(collab_vec) + (1 - g) * self.content_proj(content_vec)

# --- Module 6: DeepFM Prediction Engine ---
class DeepFMPredictionEngine(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.v = nn.Parameter(torch.randn(feature_dim, 8)) 
        
        self.mlp_tower = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, unified_features):
        # FM 1st Order (Linear)
        fm_1st = torch.sum(unified_features, dim=1, keepdim=True)
        
        # FM 2nd Order (Pairwise Interactions)
        xv = torch.matmul(unified_features, self.v)
        fm_2nd = 0.5 * torch.sum(torch.pow(xv, 2) - torch.matmul(torch.pow(unified_features, 2), torch.pow(self.v, 2)), dim=1, keepdim=True)
        
        # MLP Tower
        mlp_pred = self.mlp_tower(unified_features)
        
        # REMOVE torch.sigmoid here. Return raw logits instead.
        combined_logit = fm_1st + fm_2nd + mlp_pred
        return combined_logit #
        
# --- Main Model Architecture ---
class RecFlixAlphaModel(nn.Module):
    def __init__(self, num_critics, num_movies, vocab_size, embed_dim, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        
        # Embeddings
        self.critic_embedding = nn.Embedding(num_critics, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        
        # Modules
        self.collab_extractor = CollaborativeFeatureExtractor(embed_dim)
        self.transformer_encoder = LightweightTransformerEncoder(vocab_size, embed_dim, pad_idx)
        self.gated_fusion = GatedFusion(collab_dim=embed_dim * 2, content_dim=embed_dim, out_dim=embed_dim * 2)
        self.deep_fm = DeepFMPredictionEngine(feature_dim=embed_dim * 2)

        # Initialize core embeddings
        nn.init.normal_(self.critic_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, critic_idx, movie_idx, review_tensor):
        # Base Vectors
        user_emb = self.critic_embedding(critic_idx)
        movie_emb = self.movie_embedding(movie_idx)
        
        # Module 2: Collaborative Features
        collab_vec = self.collab_extractor(user_emb, movie_emb)
        
        # Module 4: Content/Text Features
        content_vec = self.transformer_encoder(review_tensor, user_emb, movie_emb, self.pad_idx)
        
        # Module 5: Fusion
        unified_vec = self.gated_fusion(collab_vec, content_vec)
        
        # Module 6: Prediction
        rating_pred = self.deep_fm(unified_vec)
        
        return rating_pred.squeeze(1)
