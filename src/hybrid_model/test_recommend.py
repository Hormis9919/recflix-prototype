import torch
from pathlib import Path

from src.hybrid_model.load_model import load_hybrid_model
from src.base_model.dataset import MovieLensDataset, load_movielens_ratings
from src.hybrid_model.movietext_dataset import MovieTextDataset
from src.hybrid_model.vocab import Vocabulary
from src.hybrid_model.recommend import recommend_topn


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "datasets/ml-1m"

    # Load ratings
    ratings_df = load_movielens_ratings(DATA_DIR)
    dataset = MovieLensDataset(ratings_df)

    # Load movie titles
    movies_path = DATA_DIR / "movies.dat"

    movie_id_to_title = {}

    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.strip().split("::")
            movie_id_to_title[int(movie_id)] = title

    # Load vocab
    vocab = Vocabulary(min_freq=1)
    vocab.load(ROOT / "models/hybrid_vocab.pt")

    # Load movie text dataset
    movie_text_dataset = MovieTextDataset(DATA_DIR, vocab, max_len=20)
    movie_text_tensor = movie_text_dataset.encoded_texts.to(device)

    # Load trained model
    model = load_hybrid_model(ROOT / "models/hybrid_model.pt")
    model.to(device)

    # Get recommendations
    recs = recommend_topn(
        model,
        dataset,
        movie_text_tensor,
        user_id=1,
        n=10,
        device=device
    )

    print("Top 10 Recommendations:")

    recommended_titles = [movie_id_to_title[int(movie_id)] for movie_id in recs]

    for idx, title in enumerate(recommended_titles, 1):
        print(f"{idx}. {title}")


if __name__ == "__main__":
    main()