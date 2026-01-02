import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from src.prototype.dataset import MovieLensDataset, load_movielens_ratings
from src.prototype.model import CollaborativeFilteringModel
from src.prototype.utils import get_device, set_seed
from src.prototype import config

def main():
    #Reproducability and device
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    #Load data
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "datasets" / "ml-1m"
    print("Loading ratings:")
    ratings_df = load_movielens_ratings(DATA_DIR)

    #DataSet and Loader
    dataset = MovieLensDataset(ratings_df)
    dataloader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=0)
    print(f"DataSet Size: {len(dataset)}")

    #Model
    model = CollaborativeFilteringModel(num_users=len(dataset.user2idx),num_movies=len(dataset.movie2idx),embed_dim=config.USER_EMBED_DIM).to(device)

    #Loss Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    #Training Loop
    print("Training Starts")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for user_idx, movie_idx, rating in dataloader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            #Forward Pass
            pred = model(user_idx,movie_idx)

            #Loss
            loss = criterion(pred,rating)

            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch [{epoch+1}]/{config.EPOCHS}] - Loss : {avg_loss:.4f}")
    print("Training Complete")
if __name__== "__main__":
    main()