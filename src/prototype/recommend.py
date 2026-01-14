import torch
from typing  import List

from src.prototype.model import CollaborativeFilteringModel
from src.prototype.dataset import MovieLensDataset

def recommend_topn(model: CollaborativeFilteringModel,dataset: MovieLensDataset,user_id:int,n:int=10,device: torch.device = torch.device("cpu"))->List[int]:
    #definitive function to return top n movie recommendations for a user
    model.eval()
    if user_id not in dataset.user2idx:
        raise ValueError(f"User id {user_id} not found")
    user_idx = dataset.user2idx[user_id]
    #set of movies rated by user
    rated_movies = set(dataset.ratings_df[dataset.ratings_df["user_id"]==user_id]["movie_id"].values)
    
    all_movies_ids = set(dataset.movie2idx.keys())
    candidate_movie_ids = list(all_movies_ids-rated_movies)

    #convert to indices
    movie_indices = torch.tensor([dataset.movie2idx[mid] for mid in candidate_movie_ids],dtype = torch.long, device=device)
    user_indices = torch.full((len(movie_indices),),user_idx,dtype=torch.long, device=device)
    with torch.no_grad():
        predictions = model(user_indices, movie_indices)
    
    top_n_idx = torch.topk(predictions,n).indices.tolist()
    top_n_movie_ids = [candidate_movie_ids[i] for i in top_n_idx]

    return top_n_movie_ids