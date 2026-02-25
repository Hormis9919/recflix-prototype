"""
Module: recflix_alpha.test_recommend
Purpose: Small interactive script to test recommendations locally.

This script builds the dataset and vocab, prints a sample of available
critics, prompts for a critic name, then prints the top-N movie
recommendations returned by `recommend_topn`.

It is intended as a lightweight manual test harness, not an automated
unit test.
"""

from src.recflix_alpha.recommend import recommend_topn
from pathlib import Path
from src.recflix_alpha.dataset import RecFlixAlphaDataset
from src.recflix_alpha.vocab import Vocabulary

ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = ROOT / "datasets/rotten_tomatoes_movie_reviews.csv"
MODEL_DIR = ROOT / "models"

vocab = Vocabulary(min_freq=1)
vocab.load(MODEL_DIR / "recflix_alpha_vocab.pt")

dataset = RecFlixAlphaDataset(
    reviews_path=REVIEWS_PATH,
    vocab=vocab,
    max_len=100
)

print(list(dataset.critic2idx.keys())[:20])

critic = input("Enter a name from the above list: ")
num_rec = 10
movies = recommend_topn(critic, n=num_rec)

print(f"Top {num_rec} Recommendations:")
for m in movies:
    print(m)