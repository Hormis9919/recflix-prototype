import torch
import pandas as pd
from pathlib import Path

def recommend_topn(model, dataset, static_movie_text_tensor, critic_name, data_dir: Path, n=10, device="cpu"):
    """
    Generates Top-N movie recommendations for a given Rotten Tomatoes critic.
    """
    model.eval()

    if critic_name not in dataset.critic2idx:
        raise ValueError(f"Critic '{critic_name}' not found in dataset.")

    critic_idx = dataset.critic2idx[critic_name]
    num_movies = len(dataset.movie2idx)

    critic_tensor = torch.tensor([critic_idx] * num_movies).to(device)
    movie_tensor = torch.arange(num_movies).to(device)

    # Use the static text representations for ranking
    movie_text = static_movie_text_tensor.to(device)

    with torch.no_grad():
        preds = model(critic_tensor, movie_tensor, movie_text)

    _, topn_indices = torch.topk(preds, n)
    topn_indices = topn_indices.cpu().tolist()

    # Load titles
    movies_df = pd.read_csv(data_dir / "rotten_tomatoes_movies.csv")
    id_to_title = dict(zip(movies_df["id"], movies_df["title"]))

    idx2movie = {idx: movie_id for movie_id, idx in dataset.movie2idx.items()}

    recommended_titles = []
    for idx in topn_indices:
        movie_slug = idx2movie[idx]
        title = id_to_title.get(movie_slug, movie_slug) # Fallback to slug if title missing
        recommended_titles.append(title)

    return recommended_titles
