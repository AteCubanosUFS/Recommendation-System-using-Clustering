import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # <-- Importação do t-SNE garantida aqui
import plotly.express as px        # <-- Importação do Plotly garantida aqui

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Recomendação de Filmes com IA", layout="wide", page_icon="🎬")

# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
@st.cache_data
def carregar_dados():
    movies = pd.read_csv('movies.dat', sep='::', engine='python')
    ratings = pd.read_csv('ratings.dat', sep='::', engine='python')
    
    tabela_completa = ratings.merge(movies, on='movieId')
    tabela_bons_filmes = tabela_completa[tabela_completa['rating'] >= 3.0].copy()
    
    tabela_bons_filmes['genres'] = tabela_bons_filmes['genres'].str.split('|')
    tabela_bons_filmes = tabela_bons_filmes.explode('genres')
    
    tabela_contagem = tabela_bons_filmes.groupby(['userId', 'genres'])['rating'].count().unstack().fillna(0)
    total_filmes_usuario = tabela_contagem.sum(axis=1)
    tabela_proporcao = tabela_contagem.div(total_filmes_usuario, axis=0).fillna(0)
    
    return movies, ratings, tabela_completa, tabela_proporcao

# --- TREINAMENTO DO MODELO ---
@st.cache_resource
def treinar_modelo(tabela_proporcao, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tabela_proporcao)
    
    df_clusters = tabela_proporcao.copy()
    df_clusters['Cluster'] = clusters
    return kmeans, df_clusters

# --- FUNÇÃO DO RELATÓRIO DO USUÁRIO ---
def gerar_relatorio(usuario_alvo, df_ratings, df_movies):
    avaliacoes = df_ratings[df_ratings['userId'] == usuario_alvo]
    filmes_vistos = avaliacoes.merge(df_movies, on='movieId', how='left')
    
    total = len(filmes_vistos)
    media = filmes_vistos['rating'].mean() if total > 0 else 0
    
    filmes_expandidos = filmes_vistos.copy()
    filmes_expandidos['genres'] = filmes_expandidos['genres'].str.split('|')
    filmes_expandidos = filmes_expandidos.explode('genres')
    generos_top = filmes_expandidos['genres'].value_counts()
    
    top_filmes = filmes_vistos.sort_values(by='rating', ascending=False).head(5)
    
    return total, media, generos_top, top_filmes

# --- FUNÇÃO DE RECOMENDAÇÃO ---
def recomendar_filmes(usuario_alvo, df_clusters, df_dados_originais, df_filmes, top_n=5, min_avaliacoes=3):
    if usuario_alvo not in df_clusters.index:
        return None
        
    cluster_do_usuario = df_clusters.loc[usuario_alvo, 'Cluster']
    usuarios_da_tribo = df_clusters[df_clusters['Cluster'] == cluster_do_usuario].index
    filmes_vistos_pelo_alvo = df_dados_originais[df_dados_originais['userId'] == usuario_alvo]['movieId'].unique()
    
    avaliacoes_da_tribo = df_dados_originais[df_dados_originais['userId'].isin(usuarios_da_tribo)]
    avaliacoes_validas = avaliacoes_da_tribo[~avaliacoes_da_tribo['movieId'].isin(filmes_vistos_pelo_alvo)]
    avaliacoes_unicas = avaliacoes_validas.drop_duplicates(subset=['userId', 'movieId', 'rating'])
    
    filmes_agrupados = avaliacoes_unicas.groupby(['movieId', 'title']).agg(
        nota_media_cluster=('rating', 'mean'),
        contagem_avaliacoes=('rating', 'count')
    ).reset_index()
    
    filmes_relevantes = filmes_agrupados[filmes_agrupados['contagem_avaliacoes'] >= min_avaliacoes]
    filmes_excelentes = filmes_relevantes[filmes_relevantes['nota_media_cluster'] >= 3.0]
    
    top_filmes = filmes_excelentes.sort_values(
        by=['contagem_avaliacoes', 'nota_media_cluster'], 
        ascending=[False, False]
    ).head(top_n)
    
    top_filmes_com_generos = top_filmes.merge(df_filmes[['movieId', 'genres']], on='movieId', how='left')
    top_filmes_com_generos = top_filmes_com_generos.rename(columns={
        'title': 'Título', 'genres': 'Gêneros',
        'nota_media_cluster': 'Nota Média do Grupo', 'contagem_avaliacoes': 'Qtd. Avaliações'
    })
    
    return top_filmes_com_generos[['Título', 'Gêneros', 'Nota Média do Grupo', 'Qtd. Avaliações']]

# --- FUNÇÃO PARA DETALHES DO CLUSTER ---
def obter_detalhes_cluster(cluster_alvo, df_clusters, df_ratings):
    # Quantidade de pessoas na tribo
    qtd_usuarios = len(df_clusters[df_clusters['Cluster'] == cluster_alvo])
    
    # Gêneros dominantes da tribo (Média das proporções)
    dados_do_cluster = df_clusters[df_clusters['Cluster'] == cluster_alvo].drop(columns=['Cluster'])
    top_generos_tribo = dados_do_cluster.mean().sort_values(ascending=False).head(3)
    
    # Média de filmes que as pessoas dessa tribo assistem
    usuarios_da_tribo = df_clusters[df_clusters['Cluster'] == cluster_alvo].index
    avaliacoes_tribo = df_ratings[df_ratings['userId'].isin(usuarios_da_tribo)]
    media_filmes = len(avaliacoes_tribo) / qtd_usuarios if qtd_usuarios > 0 else 0
    
    return qtd_usuarios, top_generos_tribo, media_filmes

# ==========================================
# INÍCIO DA INTERFACE (FRONT-END)
# ==========================================
st.title("🎬 Sistema de Recomendação com K-Means")

# Carrega os dados e treina o modelo
with st.spinner("Inicializando os motores da Inteligência Artificial..."):
    movies, ratings, tabela_completa, tabela_proporcao = carregar_dados()
    modelo, df_clusters = treinar_modelo(tabela_proporcao, k=5)

# --- MENU LATERAL (Sidebar) ---
st.sidebar.header("⚙️ Painel de Controle")
lista_usuarios = df_clusters.index.tolist()
usuario_selecionado = st.sidebar.selectbox("Escolha o ID do Usuário:", lista_usuarios)
qtd_rec = st.sidebar.slider("Quantos filmes recomendar?", min_value=1, max_value=15, value=5)

# Identifica o cluster do usuário selecionado
cluster_atual = df_clusters.loc[usuario_selecionado, 'Cluster']

# ==========================================
# DIVIDINDO A TELA EM 4 ABAS CLARAS
# ==========================================
aba1, aba2, aba3, aba4 = st.tabs([
    "👤 Perfil do Usuário", 
    "👥 Detalhes da Tribo", 
    "🪐 Gráficos e Mapas", 
    "🍿 Recomendações"
])

# ------------------------------------------------
# ABA 1: PERFIL DO USUÁRIO
# ------------------------------------------------
with aba1:
    st.header(f"Dados Pessoais - Usuário {usuario_selecionado}")
    st.info(f"🧠 A Inteligência Artificial classificou este usuário no **Cluster {cluster_atual}**.")
    
    total_f, media_n, generos_top, top_filmes = gerar_relatorio(usuario_selecionado, ratings, movies)
    
    col1, col2 = st.columns(2)
    col1.metric("🎬 Total de Filmes Assistidos", total_f)
    col2.metric("⭐ Média Geral de Notas", f"{media_n:.2f}")
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏆 Top Gêneros Favoritos")
        if total_f > 0:
            for genero, contagem in generos_top.head(4).items():
                porcentagem = (contagem / total_f) * 100
                st.write(f"- **{genero}**: {contagem} filmes ({porcentagem:.1f}%)")
        else:
            st.write("Nenhum dado de gênero disponível.")
            
    with col4:
        st.subheader("🥇 Top 5 Filmes com Maiores Notas")
        tabela_favoritos = top_filmes[['title', 'genres', 'rating']].rename(
            columns={'title': 'Título', 'genres': 'Gêneros', 'rating': 'Nota'}
        )
        st.dataframe(tabela_favoritos, use_container_width=True, hide_index=True)

# ------------------------------------------------
# ABA 2: DETALHES DA TRIBO (CLUSTER)
# ------------------------------------------------
with aba2:
    st.header(f"Visão Geral do Cluster {cluster_atual}")
    st.write("Entenda o comportamento das pessoas que possuem o mesmo padrão de gosto que este usuário:")
    
    # Chama a função que criamos anteriormente para os detalhes do cluster
    qtd_us, top_gens_c, med_filmes_c = obter_detalhes_cluster(cluster_atual, df_clusters, ratings)
    
    c_tribo1, c_tribo2 = st.columns(2)
    c_tribo1.metric("👥 Total de Pessoas nesta Tribo", qtd_us)
    c_tribo2.metric("🍿 Média de Filmes Assistidos (Por pessoa)", f"{med_filmes_c:.0f} filmes")
    
    st.markdown("---")
    st.subheader("DNA da Tribo (Gêneros Dominantes)")
    st.write("Esta tribo foi agrupada pela IA matematicamente por amar estes gêneros:")
    
    for gen, prop in top_gens_c.items():
        st.success(f"**{gen}** representa **{prop*100:.1f}%** da 'dieta de filmes' deste grupo.")

# ------------------------------------------------
# ABA 3: GRÁFICOS (VISÃO 3D)
# ------------------------------------------------
with aba3:
    
    col5, col6 = st.columns(2)
    
    with col5: 
        # Gráfico PCA
        pca = PCA(n_components=2)
        dados_pca = df_clusters.drop(columns=['Cluster'])
        componentes = pca.fit_transform(dados_pca)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 1. Salve o gráfico das bolinhas em uma variável (scatter_clusters)
        scatter_clusters = ax.scatter(componentes[:, 0], componentes[:, 1], c=df_clusters['Cluster'], cmap='viridis', alpha=0.5)
        
        # Destaca o usuário alvo normalmente
        idx = df_clusters.index.get_loc(usuario_selecionado)
        ax.scatter(componentes[idx, 0], componentes[idx, 1], c='red', s=200, edgecolors='black', marker='*', label='Usuário Alvo', zorder=5)
        
        # ==========================================
        # ADICIONANDO AS LEGENDAS DOS EIXOS E A GRADE
        # ==========================================
        ax.set_xlabel('Gostos Majoritários (Componente 1)', fontsize=10)
        ax.set_ylabel('Gostos Secundários (Componente 2)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # ==========================================
        # A MÁGICA DA LEGENDA DAS CORES AQUI:
        # ==========================================
        # Extrai as cores geradas e formata o texto para "Cluster 0", "Cluster 1", etc.
        handles_clusters, labels_clusters = scatter_clusters.legend_elements(fmt="Cluster {x:.0f}")
        
        # Extrai a legenda do "Usuário Alvo" (A estrela vermelha)
        handles_alvo, labels_alvo = ax.get_legend_handles_labels()
        
        # Junta as legendas dos clusters com a legenda do usuário alvo e exibe no gráfico!
        ax.legend(handles_clusters + handles_alvo, labels_clusters + labels_alvo, title="Grupos", loc="best")
        
        # Opcional: Ajusta as margens para a legenda não ficar apertada
        fig.tight_layout()
        
        st.pyplot(fig)
    
    with col6:
        st.subheader("🪐 Mapa de Clusters 3D (Visão de Profundidade)")
        st.write("Passe o mouse pelos pontos para investigar outros usuários. Gire o gráfico para ver a separação.")
        
        stats_usuarios = ratings.groupby('userId').agg(
            Total_Filmes=('rating', 'count'), Media=('rating', 'mean')
        ).reset_index()
        
        pca_3d = PCA(n_components=3)
        dados_pca = df_clusters.drop(columns=['Cluster'])
        componentes_3d = pca_3d.fit_transform(dados_pca)
        
        df_3d = pd.DataFrame(componentes_3d, columns=['Eixo X (Base)', 'Eixo Y (Variação 1)', 'Eixo Z (Variação 2)'])
        df_3d['Tribo (Cluster)'] = df_clusters['Cluster'].astype(str).apply(lambda x: f'Grupo {x}')
        df_3d['ID do Usuário'] = df_clusters.index
        
        df_3d = df_3d.merge(stats_usuarios, left_on='ID do Usuário', right_on='userId', how='left')
        df_3d['ID do Usuário'] = df_3d['ID do Usuário'].astype(str)
        df_3d['Média de Notas'] = df_3d['Media'].round(2).astype(str) + ' ⭐'
        
        fig = px.scatter_3d(
            df_3d, x='Eixo X (Base)', y='Eixo Y (Variação 1)', z='Eixo Z (Variação 2)',
            color='Tribo (Cluster)', hover_name='ID do Usuário',
            hover_data={ 
                'Total_Filmes': True, 'Média de Notas': True, 'Tribo (Cluster)': True,
                'Eixo X (Base)': False, 'Eixo Y (Variação 1)': False, 'Eixo Z (Variação 2)': False,
                'ID do Usuário': False, 'userId': False, 'Media': False
            },
            color_discrete_sequence=px.colors.qualitative.Pastel, opacity=0.8
        )
        
        idx = df_clusters.index.get_loc(usuario_selecionado)
        
        fig.add_scatter3d(
            x=[componentes_3d[idx, 0]], y=[componentes_3d[idx, 1]], z=[componentes_3d[idx, 2]],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='diamond', line=dict(color='black', width=2)),
            name='⭐️ Você', text=[f"VOCÊ (ID: {usuario_selecionado})"],
            textposition="top center", hoverinfo='skip'
        )
        
        fig.update_layout(
            legend_title_text='Grupos da IA', margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(showbackground=False, title='Afinidade Base'),
                yaxis=dict(showbackground=False, title='Variação de Gosto'),
                zaxis=dict(showbackground=False, title='Gostos Secundários')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# ABA 4: RECOMENDAÇÕES (REATIVO)
# ------------------------------------------------
with aba4:
    st.header("🍿 Por que a Inteligência Artificial recomenda?")
    st.write(f"Porque pessoas com um perfil idêntico ao seu (**Cluster {cluster_atual}**) avaliaram estes filmes com notas altíssimas:")
    
    with st.spinner('A IA está analisando os dados da tribo...'):
        recomendacoes = recomendar_filmes(
            usuario_selecionado, df_clusters, tabela_completa, movies, top_n=qtd_rec
        )
        
        if recomendacoes is not None and not recomendacoes.empty:
            st.dataframe(recomendacoes, use_container_width=True, hide_index=True)
            st.success("Recomendações geradas com sucesso!")
        else:
            st.warning("Não há recomendações novas suficientes para este usuário.")