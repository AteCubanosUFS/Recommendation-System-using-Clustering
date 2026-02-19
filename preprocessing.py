import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_DIR = "data"

OUTPUT_PATH = "processed/user_genre_matrix.csv"

def build_user_genre_matrix():

    if os.path.exists(OUTPUT_PATH):
        print("Arquivo já existe. Carregando...")
        return pd.read_csv(OUTPUT_PATH, index_col=0)

    print("Arquivo não encontrado. Gerando matriz...")

    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    ratings = ratings.drop(columns=["timestamp"])

    scaler = MinMaxScaler()
    ratings["rating"] = scaler.fit_transform(ratings[["rating"]])

    movies["genres"] = movies["genres"].str.split("|")
    movies_exploded = movies.explode("genres")

    df = ratings.merge(movies_exploded, on="movieId")

    user_genre_matrix = (
        df.groupby(["userId", "genres"])["rating"]
        .mean()
        .unstack()
        .fillna(0)
    )

    os.makedirs("processed", exist_ok=True)
    user_genre_matrix.to_csv(OUTPUT_PATH)

    print("Arquivo criado com sucesso.")
    return user_genre_matrix

if __name__ == "__main__":
    build_user_genre_matrix()
    print("Processo finalizado com sucesso")
