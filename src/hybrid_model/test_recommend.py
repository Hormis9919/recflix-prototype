import torch
from pathlib import Path

from src.hybrid_model.load_model import load_hybrid_model
from src.hybrid_model.movietext_dataset import MovieTextDataset, build_static_movie_texts
from src.hybrid_model.vocab import Vocabulary
from src.hybrid_model.recommend import recommend_topn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = Path("datasets/datasets/rotten_tomatoes")

    print("Loading vocab...")
    vocab = Vocabulary(min_freq=2)
    vocab.load(ROOT / "models/hybrid_vocab.pt")

    print("Loading dataset...")
    dataset = MovieTextDataset(DATA_DIR, vocab, max_len=100)

    print("Building static movie texts...")
    static_movie_text_tensor = build_static_movie_texts(dataset, device=device)

    print("Loading model...")
    model = load_hybrid_model(ROOT / "models/hybrid_model.pt", device=device)

    # Pick the first critic in the dictionary for testing
    target_critic = list(dataset.critic2idx.keys())[0]
    print(f"\nGenerating Top 10 Recommendations for critic: {target_critic}")

    recs = recommend_topn(
        model=model,
        dataset=dataset,
        static_movie_text_tensor=static_movie_text_tensor,
        critic_name=target_critic,
        data_dir=DATA_DIR,
        n=10,
        device=device
    )

    for idx, title in enumerate(recs, 1):
        print(f"{idx}. {title}")

if __name__ == "__main__":
    main()
