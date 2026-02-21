import pandas as pd
import streamlit as st

# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
@st.cache_data
def carregar_dados():
    
    # Carregando os dados do arquivo .dat
    tabela_movie = pd.read_csv('DataBase/movies.dat', sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin-1', header=0)
    tabela_ratings = pd.read_csv('DataBase/ratings.dat', sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'], header=0)
    
    # Joga a coluna timestamp fora para economizar memória
    tabela_ratings = tabela_ratings.drop(columns=['timestamp'])
    
    # Mergiando as tabelas lidas
    tabela_merge = tabela_ratings.merge(tabela_movie, on='movieId')
    
    # Ideia de filtro para usar os filmes que as pessoas gostam
    tabela_bons_filmes = tabela_merge[tabela_merge['rating'] >= 3.0].copy()
    
    # Separando os generos
    tabela_bons_filmes['genres'] = tabela_bons_filmes['genres'].str.split('|')
    
    # Fazendo a nova lista de generos virar linhas -> explode
    tabela_bons_filmes = tabela_bons_filmes.explode('genres')
    
    # Contando quantos filmes bons de cada gênero o usuário viu
    tabela_contagem_favoritos = tabela_bons_filmes.groupby(['userId', 'genres'])['rating'].count().unstack().fillna(0)
    
    # Transformando a contagem em PROPORÇÃO
    total_filmes_usuario = tabela_contagem_favoritos.sum(axis=1)
    tabela_proporcao = tabela_contagem_favoritos.div(total_filmes_usuario, axis=0).fillna(0)
    
    # Pega todos os IDs únicos que existiam desde o começo
    todos_os_usuarios = tabela_ratings['userId'].unique()
    
    # Força a tabela a ter todos esses IDs. 
    # Quem sumiu na filtragem ganha uma linha cheia de 0.0
    tabela_proporcao = tabela_proporcao.reindex(todos_os_usuarios, fill_value=0.0)
    
    return tabela_movie, tabela_ratings, tabela_merge, tabela_proporcao