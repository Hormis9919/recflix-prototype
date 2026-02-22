import torch
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
