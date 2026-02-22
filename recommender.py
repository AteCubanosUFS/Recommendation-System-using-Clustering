
def gerar_relatorio(usuario_alvo, df_ratings, df_movies):
    avaliacoes = df_ratings[df_ratings['userId'] == usuario_alvo]
    filmes_vistos = avaliacoes.merge(df_movies, on='movieId', how='left')
    
    total = len(filmes_vistos)
    media = filmes_vistos['rating'].mean() if total > 0 else 0
    
    filmes_expandidos = filmes_vistos.copy()
    filmes_expandidos['genres'] = filmes_expandidos['genres'].str.split('|')
    filmes_expandidos = filmes_expandidos.explode('genres')
    generos_top = filmes_expandidos['genres'].value_counts()
    
    top_filmes = filmes_vistos.sort_values(by='rating', ascending=False)
    
    return total, media, generos_top, top_filmes


# Recomenda filmes para um usuário com base no gosto do seu cluster,
# priorizando os filmes mais assistidos (populares) com as melhores notas.
def recomendar_filmes(usuario_alvo, df_clusters, df_dados_originais, df_filmes, top_n=5, min_avaliacoes=3):
    
    if usuario_alvo not in df_clusters.index:
        return None
    
    # Descobrir o cluster do usuário passado
    cluster_do_usuario = df_clusters.loc[usuario_alvo, 'Cluster']
    
    # Lista com os índices dos usuários(id) do mesmo cluster
    usuarios_do_cluster = df_clusters[df_clusters['Cluster'] == cluster_do_usuario].index
    
    # Lista com os filmes vistos pelo usuários indo pegar na tabela com todos os dados
    filmes_vistos_pelo_alvo = df_dados_originais[df_dados_originais['userId'] == usuario_alvo]['movieId'].unique()
    
    avaliacoes_do_cluster = df_dados_originais[df_dados_originais['userId'].isin(usuarios_do_cluster)]
    
    # Pego as avaliações dos filmes que o usuário não viu
    avaliacoes_validas = avaliacoes_do_cluster[~avaliacoes_do_cluster['movieId'].isin(filmes_vistos_pelo_alvo)]
    avaliacoes_unicas = avaliacoes_validas.drop_duplicates(subset=['userId', 'movieId', 'rating'])
    
    # Media e quantas pessoas avaliaram os filmes
    filmes_agrupados = avaliacoes_unicas.groupby(['movieId', 'title']).agg(
        nota_media_cluster=('rating', 'mean'),
        contagem_avaliacoes=('rating', 'count')
    ).reset_index() # volta para ser representado em colunas
    
    filmes_relevantes = filmes_agrupados[filmes_agrupados['contagem_avaliacoes'] >= min_avaliacoes]
    filmes_excelentes = filmes_relevantes[filmes_relevantes['nota_media_cluster'] >= 3.0]
    
    # Ordenamos os excelentes pelos mais populares e pegamos o top_n
    top_filmes = filmes_excelentes.sort_values(
        by=['contagem_avaliacoes', 'nota_media_cluster'], 
        ascending=[False, False]
    ).head(top_n)
    
    # JoinLeft das tabelas para trazer mais informação
    top_filmes_com_generos = top_filmes.merge(df_filmes[['movieId', 'genres']], on='movieId', how='left')
    
    # Renomeando as colunas
    top_filmes_com_generos = top_filmes_com_generos.rename(columns={
        'title': 'Título', 'genres': 'Gêneros',
        'nota_media_cluster': 'Nota Média do cluster', 'contagem_avaliacoes': 'Qtd. Avaliações'
    })
    
    return top_filmes_com_generos[['Título', 'Gêneros', 'Nota Média do cluster', 'Qtd. Avaliações']]

def obter_detalhes_cluster(cluster_alvo, df_clusters, df_ratings):
    qtd_usuarios = len(df_clusters[df_clusters['Cluster'] == cluster_alvo])
    dados_do_cluster = df_clusters[df_clusters['Cluster'] == cluster_alvo].drop(columns=['Cluster'])
    top_generos_cluster = dados_do_cluster.mean().sort_values(ascending=False).head(3)
    
    usuarios_da_cluster = df_clusters[df_clusters['Cluster'] == cluster_alvo].index
    avaliacoes_cluster = df_ratings[df_ratings['userId'].isin(usuarios_da_cluster)]
    media_filmes = len(avaliacoes_cluster) / qtd_usuarios if qtd_usuarios > 0 else 0
    
    return qtd_usuarios, top_generos_cluster, media_filmes

# Gerar as personas do cluster
def gerar_descricao_cluster(top_generos):
    if top_generos.empty:
        return "Cluster Indefinido", "Dados insuficientes para mapear o perfil de consumo."
        
    genero_principal = top_generos.index[0]
    
    dicionario_personas = {
        'Action': "Perfil Adrenalina",
        'Comedy': "Perfil Entretenimento",
        'Drama': "Perfil Reflexivo",
        'Horror': "Perfil Suspense e Terror",
        'Romance': "Perfil Romântico",
        'Sci-Fi': "Perfil Ficção Científica",
        'Adventure': "Perfil Aventura",
        'Fantasy': "Perfil Fantasia",
        'Animation': "Perfil Animação",
        'Documentary': "Perfil Informativo",
        'Thriller': "Perfil Investigativo",
        'Crime': "Perfil Policial"
    }
    
    titulo = dicionario_personas.get(genero_principal, "Perfil Eclético")
    
    descricao = f"Grupo caracterizado pela preferência majoritária em filmes de {genero_principal}."
    
    # Colocando os outros gêneros relevantes
    outros_validos = top_generos[top_generos > 0].index[1:].tolist()
    if outros_validos:
        descricao += f" Apresenta também engajamento relevante nos gêneros: {', '.join(outros_validos)}."
        
    return titulo, descricao