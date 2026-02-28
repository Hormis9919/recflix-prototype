import torch
from src.recflix_alpha.model import RecFlixAlphaModel

def load_recflix_alpha_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = RecFlixAlphaModel(
        num_critics=checkpoint["num_critics"],
        num_movies=checkpoint["num_movies"],
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        pad_idx=checkpoint["pad_idx"],
    ).to(device)

    # Handles the '_orig_mod.' prefix added by torch.compile
    state_dict = checkpoint["model_state_dict"]
    fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(fixed_state_dict)
    model.eval()
    return model
