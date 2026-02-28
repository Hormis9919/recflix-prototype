import pandas as pd
import numpy as np

def generate_negative_samples(positive_df, all_movies_list, num_negatives=4):
    """
    Generates negative samples for a recommendation dataset.
    
    Args:
        positive_df: DataFrame containing 'user_id', 'movie_id', 'label' (all > 0)
        all_movies_list: A list or set of all possible movie_ids in the system.
        num_negatives: The ratio of negatives to generate per positive interaction.
                       (e.g., 4 means 4 negative samples for every 1 positive).
                       
    Returns:
        A combined DataFrame of shuffled positive and negative interactions.
    """
    print(f"Generating negative samples (Target Ratio 1:{num_negatives})...")
    
    all_movie_ids = set(all_movies_list)
    
    # 1. Map out exactly what movies each user HAS interacted with
    user_histories = positive_df.groupby('user_id')['movie_id'].apply(set).to_dict()
    
    negative_records = []
    
    # 2. Iterate through users and sample unseen movies
    for user_id, seen_movies in user_histories.items():
        # Find all movies this user hasn't seen
        unseen_movies = list(all_movie_ids - seen_movies)
        
        # Calculate how many negatives we want for this user based on their positive count
        target_n_samples = len(seen_movies) * num_negatives
        
        # Make sure we don't try to sample more movies than actually exist
        n_samples = min(target_n_samples, len(unseen_movies))
        
        if n_samples > 0:
            # Randomly pick 'n_samples' from the unseen list
            sampled_negatives = np.random.choice(unseen_movies, size=n_samples, replace=False)
            
            for movie_id in sampled_negatives:
                negative_records.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'reviewText': "",       # Empty because it's an implicit non-action
                    'label': 0.0,           # The crucial negative label
                    'data_type': 'implicit_negative'
                })
                
    # 3. Combine and Shuffle
    neg_df = pd.DataFrame(negative_records)
    combined_df = pd.concat([positive_df, neg_df], ignore_index=True)
    
    # Shuffle the dataset so the model doesn't learn in chunks of 1s then 0s
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f" -> Added {len(neg_df)} negative samples.")
    print(f" -> Total dataset size ready for training: {len(combined_df)}")
    
    return combined_df

# --- Quick Test Block ---
if __name__ == "__main__":
    # Mock data to test the logic
    mock_positives = pd.DataFrame({
        'user_id': ['CriticA', 'CriticA', 'CriticB'],
        'movie_id': ['Movie1', 'Movie2', 'Movie1'],
        'reviewText': ['Loved it', 'Okay', 'Great'],
        'label': [1.0, 0.8, 0.9],
        'data_type': ['explicit', 'explicit', 'explicit']
    })
    
    all_movies = ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5']
    
    final_df = generate_negative_samples(mock_positives, all_movies, num_negatives=2)
    print("\nSample Output:")
    print(final_df.head(10))
