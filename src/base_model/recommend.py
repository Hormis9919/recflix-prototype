import torch
from typing import List

from src.base_model.model import CollaborativeFilteringModel
from src.base_model.dataset import RTRatingsDataset

def recommend_topn(
    model: CollaborativeFilteringModel,
    dataset: RTRatingsDataset,
    critic_name: str,
    n: int = 10,
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """Returns top N movie ID recommendations for a specific Rotten Tomatoes critic."""
    model.eval()
    
    # Check if the critic exists in our parsed dataset
    if critic_name not in dataset.critic2idx:
        raise ValueError(f"Critic '{critic_name}' not found in the dataset.")
        
    critic_idx = dataset.critic2idx[critic_name]
    
    # Set of movie IDs already rated by this critic (using string 'id' and 'criticName')
    rated_movies = set(dataset.ratings_df[dataset.ratings_df["criticName"] == critic_name]["id"].values)
    
    # Identify unseen movies
    all_movies_ids = set(dataset.movie2idx.keys())
    candidate_movie_ids = list(all_movies_ids - rated_movies)

    # Convert candidates to integer indices for the embedding layers
    movie_indices = torch.tensor(
        [dataset.movie2idx[mid] for mid in candidate_movie_ids], 
        dtype=torch.long, 
        device=device
    )
    user_indices = torch.full((len(movie_indices),), critic_idx, dtype=torch.long, device=device)
    
    with torch.no_grad():
        predictions = model(user_indices, movie_indices)
    
    # Retrieve top K indices and map them back to original Rotten Tomatoes string IDs
    top_n_idx = torch.topk(predictions, n).indices.tolist()
    top_n_movie_ids = [candidate_movie_ids[i] for i in top_n_idx]

    return top_n_movie_ids