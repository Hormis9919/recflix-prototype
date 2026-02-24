import torch
import math


def rmse(preds, targets):
    """
    Root Mean Squared Error for rating prediction.
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def hit_ratio_at_k(model, dataset, device, k=10, threshold=4.0):
    """
    Computes Hit Ratio@K.
    Checks whether at least one relevant movie appears in top-K.
    """
    model.eval()
    hits = 0
    total_users = len(dataset.user2idx)

    with torch.no_grad():
        for user_idx in dataset.user2idx.values():

            # Predict scores for all movies for this user
            user_tensor = torch.tensor([user_idx] * len(dataset.movie2idx)).to(device)
            movie_tensor = torch.arange(len(dataset.movie2idx)).to(device)

            preds = model(user_tensor, movie_tensor)

            # Get Top-K movie indices
            _, topk_indices = torch.topk(preds, k)
            topk_indices = topk_indices.cpu().tolist()

            # Get relevant movies from validation data
            user_data = dataset.ratings_df
            relevant_movies = user_data[
                (user_data["user_idx"] == user_idx) &
                (user_data["rating"] >= threshold)
            ]["movie_idx"].tolist()

            # Check hit
            if any(movie in relevant_movies for movie in topk_indices):
                hits += 1

    return hits / total_users


def ndcg_at_k(model, dataset, device, k=10, threshold=4.0):
    """
    Computes NDCG@K.
    Measures ranking quality.
    """
    model.eval()
    total_ndcg = 0
    total_users = len(dataset.user2idx)

    with torch.no_grad():
        for user_idx in dataset.user2idx.values():

            user_tensor = torch.tensor([user_idx] * len(dataset.movie2idx)).to(device)
            movie_tensor = torch.arange(len(dataset.movie2idx)).to(device)

            preds = model(user_tensor, movie_tensor)

            _, topk_indices = torch.topk(preds, k)
            topk_indices = topk_indices.cpu().tolist()

            user_data = dataset.ratings_df
            relevant_movies = user_data[
                (user_data["user_idx"] == user_idx) &
                (user_data["rating"] >= threshold)
            ]["movie_idx"].tolist()

            # Compute DCG
            dcg = 0
            for rank, movie in enumerate(topk_indices):
                if movie in relevant_movies:
                    dcg += 1 / math.log2(rank + 2)

            # Compute Ideal DCG
            ideal_dcg = sum(
                1 / math.log2(i + 2)
                for i in range(min(len(relevant_movies), k))
            )

            if ideal_dcg > 0:
                total_ndcg += dcg / ideal_dcg

    return total_ndcg / total_users