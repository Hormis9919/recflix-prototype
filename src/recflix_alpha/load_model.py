"""
Module: recflix_alpha.load_model
Purpose: Convenience loader that reconstructs and returns a trained model.

This module exposes `load_recflix_alpha_model(model_path, device)` which
loads a PyTorch checkpoint saved by `train.py`, instantiates the
`RecFlixAlphaModel` with the saved hyperparameters and weights, moves
it to the requested device, and sets it to evaluation mode.

Responsibilities:
- Centralize checkpoint deserialization logic and model reconstruction.
"""

import torch
from src.recflix_alpha.model import RecFlixAlphaModel


def load_recflix_alpha_model(model_path, device="cpu"):

    checkpoint = torch.load(model_path, map_location=device)

    model = RecFlixAlphaModel(
        num_critics=checkpoint["num_critics"],
        num_movies=checkpoint["num_movies"],
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        pad_idx=checkpoint["pad_idx"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model