"""
Module: recflix_alpha.review_encoder
Purpose: Convert token sequences to fixed-size review vectors.

`ReviewEncoder` is intentionally simple: it uses an Embedding layer
followed by mask-aware mean pooling to produce a single vector per
review. This representation is used by `RecFlixAlphaModel` alongside
critic and movie embeddings.

Design notes:
- Embedding + mean pooling is fast and robust for short reviews.
- The encoder exposes a small, reusable interface: `forward(text)`.
"""

import torch
import torch.nn as nn


class ReviewEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

    def forward(self, text):

        # Step 1: Embed tokens
        # (batch_size, seq_len, embed_dim)
        embeddings = self.embedding(text)

        # Step 2: Create mask for PAD tokens
        # PAD tokens have embedding index = pad_idx
        mask = (text != self.embedding.padding_idx).unsqueeze(-1)

        # Step 3: Zero out PAD embeddings
        masked_embeddings = embeddings * mask

        # Step 4: Sum embeddings along sequence dimension
        summed = masked_embeddings.sum(dim=1)

        # Step 5: Count non-pad tokens per sample
        lengths = mask.sum(dim=1).clamp(min=1)

        # Step 6: Mean pooling
        mean_pooled = summed / lengths

        return mean_pooled
