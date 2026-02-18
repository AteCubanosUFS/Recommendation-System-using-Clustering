import pandas as pd
import numpy as np
import os


def load_data():
    """Carrega os arquivos da pasta data"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print("Dados carregados com sucesso.")
    return movies, ratings, BASE_DIR

def prepare_data(movies, ratings):
    """Realiza merge, limpeza e criação da matriz Usuário x Gênero"""

    df = ratings.merge(movies, on="movieId")

    df = df[df['genres'] != '(no genres listed)']

    df['genres'] = df['genres'].str.split('|')
    df_exploded = df.explode('genres')

    genre_dummies = pd.get_dummies(df_exploded['genres'])

    df_final = pd.concat(
        [df_exploded[['userId', 'rating']], genre_dummies],
        axis=1
    )

    for genre in genre_dummies.columns:
        df_final[genre] = df_final[genre] * df_final['rating']

    user_sum = df_final.groupby('userId')[genre_dummies.columns].sum()

    user_count = genre_dummies.groupby(df_exploded['userId']).sum()

    user_genre_matrix = user_sum.divide(user_count.replace(0, np.nan))

    user_genre_matrix = user_genre_matrix.fillna(0)


    print("Matriz Usuário x Gênero criada com sucesso.")
    print(f"Dimensão da matriz: {user_genre_matrix.shape}")

    return user_genre_matrix


def save_matrix(user_genre_matrix, base_dir):
    """Salva a matriz em CSV"""
    output_path = os.path.join(base_dir, "user_genre_matrix.csv")
    user_genre_matrix.to_csv(output_path)
    print(f"Matriz salva em: {output_path}")

if __name__ == "__main__":
    movies, ratings, BASE_DIR = load_data()
    user_genre_matrix = prepare_data(movies, ratings)
    save_matrix(user_genre_matrix, BASE_DIR)