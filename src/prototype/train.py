import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from src.prototype.dataset import MovieLensDataset, load_movielens_ratings
from src.prototype.model import CollaborativeFilteringModel
from src.prototype.utils import get_device, set_seed
from src.prototype.evaluation import rmse, mae
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
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=0)
    val_loader = DataLoader(val_dataset,batch_size=config.BATCH_SIZE,shuffle=False,num_workers=0)
    print(f"Train Size: {train_size}, Validation Size: {val_size}")

    #Model
    model = CollaborativeFilteringModel(num_users=len(dataset.user2idx),num_movies=len(dataset.movie2idx),embed_dim=config.USER_EMBED_DIM).to(device)

    #Loss Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    #Training Loop
    print("Training Starts")
    for epoch in range(config.EPOCHS):
        #training
        model.train()
        total_loss = 0.0
        for user_idx, movie_idx, rating in train_loader:
            user_idx =user_idx.to(device)
            movie_idx = movie_idx.to(device)
            rating = rating.to(device)

            pred = model(user_idx,movie_idx)
            loss = criterion(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
        avg_train_los = total_loss/len(train_loader)
        #validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for user_idx, movie_idx, rating in val_loader:
                user_idx = user_idx.to(device)
                movie_idx = movie_idx.to(device)
                rating = rating.to(device)

                pred = model(user_idx,movie_idx)

                all_preds.append(pred)
                all_targets.append(rating)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_rmse = rmse(all_preds,all_targets)
        val_mae = mae(all_preds,all_targets)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}]\nTrain Loss: {avg_train_los:.4f}\nVal RMSE: {val_rmse:.4f}\nVal MAE: {val_mae:.4f}")
    #Saving trained model
    MODEL_DIR = ROOT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "cf_model.pt"
    torch.save(model.state_dict(),model_path)
    print(f"Model Saved to {model_path}")
    print("Training Complete")
if __name__== "__main__":
    main()