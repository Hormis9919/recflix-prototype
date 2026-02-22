import torch 
import math

def rmse(predictions: torch.Tensor, targets: torch.Tensor) ->float:
    return math.sqrt(torch.mean((predictions-targets)**2).item())
#focus on large errors, common in recommender systems
def mae(predictions: torch.Tensor,targets:torch.Tensor) -> float:
    return torch.mean(torch.abs(predictions-targets)).item()
#easy to interpret, robust to outliers
