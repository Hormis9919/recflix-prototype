import torch
import math

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def hit_ratio_at_k(model, dataset, static_movie_text_tensor, device, k=10, threshold=0.8):
    model.eval()
    hits = 0
    total_critics = len(dataset.critic2idx)
    
    df = dataset.df.copy()
    df['critic_idx'] = df['criticName'].map(dataset.critic2idx)
    df['movie_idx'] = df['id'].map(dataset.movie2idx)

    with torch.no_grad():
        for critic_idx in dataset.critic2idx.values():
            num_movies = len(dataset.movie2idx)

            critic_tensor = torch.tensor([critic_idx] * num_movies).to(device)
            movie_tensor = torch.arange(num_movies).to(device)
            movie_text = static_movie_text_tensor.to(device)

            preds = model(critic_tensor, movie_tensor, movie_text)

            _, topk_indices = torch.topk(preds, k)
            topk_indices = topk_indices.cpu().tolist()

            relevant_movies = df[
                (df["critic_idx"] == critic_idx) &
                (df["parsed_score"] >= threshold)
            ]["movie_idx"].tolist()

            if any(movie in relevant_movies for movie in topk_indices):
                hits += 1

    return hits / total_critics

def ndcg_at_k(model, dataset, static_movie_text_tensor, device, k=10, threshold=0.8):
    model.eval()
    total_ndcg = 0
    total_critics = len(dataset.critic2idx)
    
    df = dataset.df.copy()
    df['critic_idx'] = df['criticName'].map(dataset.critic2idx)
    df['movie_idx'] = df['id'].map(dataset.movie2idx)

    with torch.no_grad():
        for critic_idx in dataset.critic2idx.values():
            num_movies = len(dataset.movie2idx)

            critic_tensor = torch.tensor([critic_idx] * num_movies).to(device)
            movie_tensor = torch.arange(num_movies).to(device)
            movie_text = static_movie_text_tensor.to(device)

            preds = model(critic_tensor, movie_tensor, movie_text)

            _, topk_indices = torch.topk(preds, k)
            topk_indices = topk_indices.cpu().tolist()

            relevant_movies = df[
                (df["critic_idx"] == critic_idx) &
                (df["parsed_score"] >= threshold)
            ]["movie_idx"].tolist()

            dcg = 0
            for rank, movie in enumerate(topk_indices):
                if movie in relevant_movies:
                    dcg += 1 / math.log2(rank + 2)

            ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_movies), k)))

            if ideal_dcg > 0:
                total_ndcg += dcg / ideal_dcg

    return total_ndcg / total_critics
