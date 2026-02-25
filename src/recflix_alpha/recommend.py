"""
Module: recflix_alpha.recommend
Purpose: Produce top-N movie recommendations for a given critic.

This module exposes `recommend_topn(critic_name, n)` which:
- Loads the model and vocab, builds the dataset.
- For a requested critic, scores every unique movie using an available
  review text (first seen review for each movie) and the trained model.
- Returns the top-N movie titles using the Rotten Tomatoes metadata CSV.

Notes:
- This is a simple scoring-based recommender (score all candidates and
  take the top-N) and uses available review text as the textual context
  for each movie. It assumes the dataset and model artifacts are present
  in `models/` and movie metadata in `datasets/`.
"""

import torch
from pathlib import Path
import pandas as pd

from src.recflix_alpha.dataset import RecFlixAlphaDataset
from src.recflix_alpha.vocab import Vocabulary
from src.recflix_alpha.load_model import load_recflix_alpha_model


def recommend_topn(critic_name, n=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).resolve().parents[2]
    REVIEWS_PATH = ROOT / "datasets/rotten_tomatoes_movie_reviews.csv"
    MODEL_DIR = ROOT / "models"

    # Load vocab
    vocab = Vocabulary(min_freq=1)
    vocab.load(MODEL_DIR / "recflix_alpha_vocab.pt")

    # Load dataset
    dataset = RecFlixAlphaDataset(
        reviews_path=REVIEWS_PATH,
        vocab=vocab,
        max_len=100
    )

    # Load movie metadata for mapping id to name
    MOVIES_PATH = ROOT / "datasets/rotten_tomatoes_movies.csv"
    movies_df = pd.read_csv(MOVIES_PATH)

    id_to_title = dict(zip(movies_df["id"], movies_df["title"]))

    # Load model
    model = load_recflix_alpha_model(
        MODEL_DIR / "recflix_alpha.pt",
        device=device
    )

    # Check critic exists
    if critic_name not in dataset.critic2idx:
        print("Critic not found.")
        return []

    critic_idx = dataset.critic2idx[critic_name]

    # Get all unique movies
    movie_indices = list(dataset.movie2idx.values())

    critic_tensor = torch.tensor([critic_idx] * len(movie_indices)).to(device)
    movie_tensor = torch.tensor(movie_indices).to(device)

    # For recommendation we need review text for each movie
    # Use the first review text instance per movie
    movie_review_map = {}

    for i in range(len(dataset)):
        c_idx, m_idx, review_tensor, score = dataset[i]
        if int(m_idx) not in movie_review_map:
            movie_review_map[int(m_idx)] = review_tensor

    review_tensor_list = [
        movie_review_map[m] for m in movie_indices
    ]

    review_tensor = torch.stack(review_tensor_list).to(device)

    with torch.no_grad():
        preds = model(critic_tensor, movie_tensor, review_tensor)

    top_indices = torch.topk(preds, n).indices.cpu().tolist()

    # Reverse movie2idx to get titles
    idx2movie = {idx: movie for movie, idx in dataset.movie2idx.items()}

    recommended_movies = []

    for i in top_indices:
        movie_slug = idx2movie[movie_indices[i]]
        title = id_to_title.get(movie_slug, movie_slug)
        recommended_movies.append(title)

    return recommended_movies