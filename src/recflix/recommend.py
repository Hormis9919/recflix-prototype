import torch
from pathlib import Path
import pandas as pd

from src.recflix.dataset import UnifiedRecFlixDataset
from src.recflix.vocab import Vocabulary
from src.recflix.load_model import load_recflix_model

# Enable TF32 for inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def recommend_topn(critic_name, n=10, dataset=None, favorite_movie_titles=None):
    if favorite_movie_titles is None:
        favorite_movie_titles = []
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = Path(__file__).resolve().parents[2]
    REVIEWS_PATH = ROOT / "datasets/datasets/rotten_tomatoes/rotten_tomatoes_movie_reviews.csv"
    MOVIES_PATH = ROOT / "datasets/datasets/rotten_tomatoes/rotten_tomatoes_movies.csv"
    MODEL_DIR = ROOT / "models"

    vocab = Vocabulary(min_freq=5)
    vocab.load(MODEL_DIR / "recflix_vocab.pt")

    if dataset is None:
        dataset = UnifiedRecFlixDataset(
            explicit_path=REVIEWS_PATH, 
            vocab=vocab,
            max_len=50 
        )

    movies_df = pd.read_csv(MOVIES_PATH)
    id_to_title = dict(zip(movies_df["id"], movies_df["title"]))
    title_to_id = {str(v).lower().strip(): k for k, v in id_to_title.items()}

    model = load_recflix_model(
        MODEL_DIR / "recflix.pt",
        device=device
    )

    movie_indices = list(dataset.movie2idx.values())
    movie_tensor = torch.tensor(movie_indices).to(device)

    is_new_user = False
    if critic_name not in dataset.critic2idx:
        print(f"\n[!] Cold-Start Triggered: '{critic_name}' is a new user.")
        is_new_user = True
    else:
        critic_idx = dataset.critic2idx[critic_name]

    # Pre-calculate movie proxy texts
    movie_review_map = {}
    for i in range(len(dataset)):
        _, m_idx, review_tensor, _ = dataset[i]
        m_idx_val = int(m_idx)
        if m_idx_val not in movie_review_map:
            movie_review_map[m_idx_val] = review_tensor

    review_tensor_batch = torch.stack([movie_review_map[m] for m in movie_indices]).to(device)

    model.eval()
    all_preds = []
    # OPTIMIZATION: Small batch size to keep Transformer attention fast and lean
    batch_size = 512 

    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            # Cold-start user profile generation
            if is_new_user:
                mean_user_emb = model.critic_embedding.weight.mean(dim=0, keepdim=True)
                if favorite_movie_titles:
                    fav_ids = [title_to_id[t.strip().lower()] for t in favorite_movie_titles if t.strip().lower() in title_to_id]
                    fav_movie_idxs = [dataset.movie2idx[m_id] for m_id in fav_ids if m_id in dataset.movie2idx]
                    if fav_movie_idxs:
                        mask = torch.isin(dataset.movie_indices, torch.tensor(fav_movie_idxs)) & (dataset.labels > 0.7)
                        cohort_critics = dataset.critic_indices[mask].unique()
                        if len(cohort_critics) > 0:
                            print(f" -> Found {len(cohort_critics)} critics for cohort averaging...")
                            mean_user_emb = model.critic_embedding(cohort_critics.to(device)).mean(dim=0, keepdim=True)

            # Batched Prediction Loop
            for i in range(0, len(movie_indices), batch_size):
                b_movie_tensor = movie_tensor[i : i + batch_size]
                b_review_tensor = review_tensor_batch[i : i + batch_size]
                curr_batch_len = b_movie_tensor.size(0)

                if is_new_user:
                    user_emb = mean_user_emb.expand(curr_batch_len, -1)
                    movie_emb = model.movie_embedding(b_movie_tensor)
                    
                    collab_vec = model.collab_extractor(user_emb, movie_emb)
                    content_vec = model.transformer_encoder(b_review_tensor, user_emb, movie_emb, model.pad_idx)
                    unified_vec = model.gated_fusion(collab_vec, content_vec)
                    
                    logits = model.deep_fm(unified_vec).squeeze(1)
                else:
                    b_critic_tensor = torch.tensor([critic_idx] * curr_batch_len).to(device)
                    logits = model(b_critic_tensor, b_movie_tensor, b_review_tensor)
                
                all_preds.append(torch.sigmoid(logits))

    preds = torch.cat(all_preds)
    top_indices = torch.topk(preds, n).indices.cpu().tolist()
    idx2movie = {idx: movie for movie, idx in dataset.movie2idx.items()}

    recommended_movies = []
    for i in top_indices:
        movie_slug = idx2movie[movie_indices[i]]
        title = id_to_title.get(movie_slug, movie_slug)
        recommended_movies.append(title)

    return recommended_movies
