import torch


def recommend_topn(model, dataset, movie_text_tensor, user_id, n=10, device="cpu"):
    """
    Generates Top-N movie recommendations for a given user.
    """

    model.eval()

    # Convert user_id to internal index
    if user_id not in dataset.user2idx:
        raise ValueError("User ID not found in dataset.")

    user_idx = dataset.user2idx[user_id]

    # Create tensors for all movies
    num_movies = len(dataset.movie2idx)

    user_tensor = torch.tensor([user_idx] * num_movies).to(device)
    movie_tensor = torch.arange(num_movies).to(device)

    # Get corresponding movie text embeddings
    movie_text = movie_text_tensor[movie_tensor].to(device)

    with torch.no_grad():
        preds = model(user_tensor, movie_tensor, movie_text)

    # Get Top-N movie indices
    _, topn_indices = torch.topk(preds, n)

    # Convert indices back to original movie IDs
    idx2movie = {idx: movie_id for movie_id, idx in dataset.movie2idx.items()}

    recommended_movie_ids = [
        idx2movie[idx.item()] for idx in topn_indices
    ]

    return recommended_movie_ids
