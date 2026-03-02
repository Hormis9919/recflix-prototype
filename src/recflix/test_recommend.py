import os
import pickle
import warnings

# Suppress PyTorch weight loading warnings for a clean presentation terminal
warnings.filterwarnings("ignore", category=FutureWarning)

from src.recflix.recommend import recommend_topn
from pathlib import Path
from src.recflix.dataset import UnifiedRecFlixDataset
from src.recflix.vocab import Vocabulary

ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = ROOT / "datasets/datasets/rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"
MODEL_DIR = ROOT / "models"

# 1. Load Vocab once
vocab = Vocabulary(min_freq=5)
vocab.load(MODEL_DIR / "recflix_vocab.pt")

# 2. Optimized Dataset Loading with Cache Check
CACHE_PATH = ROOT / "cache/recflix_dataset_cache.pkl"

print("Starting fast dataset boot...")
if CACHE_PATH.exists():
    # BOOT TIME: ~5 Seconds
    dataset = UnifiedRecFlixDataset.load(CACHE_PATH)
else:
    # BOOT TIME: ~1-2 Minutes (Optimized Vectorized Run)
    print("No cache found. Processing CSV (First-time only)...")
    dataset = UnifiedRecFlixDataset(
        explicit_path=REVIEWS_PATH,
        vocab=vocab,
        max_len=50 
    )
    # Saving the processed dataset for future runs
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(CACHE_PATH)

def run_interactive_test():
    print("\n--- Available Existing Critics (Sample) ---")
    print(list(dataset.critic2idx.keys())[:10])
    print("-" * 43)

    critic = input("Enter an existing name from above, OR type a brand new name: ")

    favorite_movies = []
    if critic not in dataset.critic2idx:
        print(f"\nWelcome to RecFlix, {critic}!")
        print("To help us personalize your feed, please enter up to 3 of your favorite movies.")
        user_input = input("Movies (e.g., Inception, Interstellar): ")
        
        if user_input.strip():
            favorite_movies = [m.strip() for m in user_input.split(',')]
            favorite_movies = favorite_movies[:3]

    # Pass the ALREADY LOADED dataset object to prevent re-initialization
    movies = recommend_topn(critic, n=10, dataset=dataset, favorite_movie_titles=favorite_movies)

    print(f"\nTop Recommendations for {critic}:")
    for m in movies:
        print(f"- {m}")

if __name__ == "__main__":
    while True:
        run_interactive_test()
        cont = input("\nRun another test? (y/n): ")
        if cont.lower() != 'y':
            break