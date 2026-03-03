import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.base_model.dataset import RTRatingsDataset, load_rt_reviews
from src.base_model.model import CollaborativeFilteringModel
from src.common_files.utils import get_device, set_seed
from src.base_model import config

def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    ROOT_DIR = Path(__file__).resolve().parents[3]
    # Update this to wherever your RT files are located
    DATA_DIR = ROOT_DIR / "datasets/datasets" / "rotten_tomatoes" 
    print("Loading ratings...")
    ratings_df = load_rt_reviews(DATA_DIR)

    # DataSet and Loader
    dataset = RTRatingsDataset(ratings_df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Train Size: {train_size}, Validation Size: {val_size}")

    # Model
    model = CollaborativeFilteringModel(
        num_users=len(dataset.critic2idx), 
        num_movies=len(dataset.movie2idx), 
        embed_dim=config.USER_EMBED_DIM
    ).to(device)

    # Loss Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training Loop
    print("Training Starts")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for user_idx, movie_idx, rating in train_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            pred = model(user_idx, movie_idx)
            loss = criterion(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}]\nTrain Loss: {avg_train_loss:.4f}\n")
        
    # Saving trained model
    MODEL_DIR = ROOT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "cf_model_rt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_users": len(dataset.critic2idx),
        "num_movies": len(dataset.movie2idx),
        "embed_dim": config.USER_EMBED_DIM,
    }, model_path)
    print(f"Model Saved to {model_path}")

if __name__== "__main__":
    main()