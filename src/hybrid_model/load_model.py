import torch
from src.hybrid_model.hybrid_model import HybridRecommender


def load_hybrid_model(model_path, device="cpu"):
    """
    Load a HybridRecommender checkpoint.
    NOTE: The bare  model = load_hybrid_model(...)  line that used to
    exist at module level has been removed — it executed on every import
    and crashed whenever the .pt file did not exist yet.
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = HybridRecommender(
        num_users  = checkpoint["num_users"],
        num_movies = checkpoint["num_movies"],
        vocab_size = checkpoint["vocab_size"],
        embed_dim  = checkpoint["embed_dim"],
        pad_idx    = checkpoint["pad_idx"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model