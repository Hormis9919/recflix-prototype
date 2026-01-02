import pandas as pd
from pathlib import Path
#to navigate within the system

#first we find root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

#create dataset address
DATA_DIR = ROOT_DIR / "datasets" / "ml-1m"

ratings_path = DATA_DIR / "ratings.dat"
movies_path = DATA_DIR / "movies.dat"
users_path = DATA_DIR / "users.dat"

#loading ratings, movies, and users
ratings = pd.read_csv(ratings_path,sep="::",engine="python",names=["user_id","movie_id","rating","timestamp"])
movies = pd.read_csv(movies_path,sep="::",engine="python",names=["movie_id","title","genre"],encoding="latin-1")
users = pd.read_csv(users_path,sep="::",engine="python",names=["user_id","gender","age","occupation","zip_code"])

#sample data
print("Ratings",ratings.head(),sep="\n")
print("Movies",movies.head(),sep="\n")
print("Users",users.head(),sep="\n")

#Checking uqiue entries
print("users: ",ratings.user_id.nunique())
print("movies: ",ratings.movie_id.nunique())
print("number of ratings: ",len(ratings))

    