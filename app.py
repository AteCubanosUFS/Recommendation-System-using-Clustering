import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Recomendação de Filmes com K-Means", 
    layout="wide", 
    page_icon="🎬"
)

# --- CONFIGURAÇÃO DE CORES (EXPANDIDA PARA 25 CLUSTERS) ---
PALETA_CLUSTERS = {
    0: '#636EFA',  # Azul
    1: '#00CC96',  # Verde
    2: '#AB63FA',  # Roxo
    3: '#FFA15A',  # Laranja
    4: '#19D3F3',  # Ciano
    5: '#E763FA',  # Magenta
    6: '#FECB52',  # Amarelo
    7: '#8C564B',  # Marrom
    8: '#FF6692',  # Rosa Claro
    9: '#B6E880',  # Verde Claro
    10: '#FF97FF', # Rosa Orquídea
    11: '#191970', # Azul Marinho
    12: '#32CD32', # Lima
    13: '#FFD700', # Ouro
    14: '#CD5C5C', # Vermelho Indiano (Tom terroso)
    15: '#4B0082', # Índigo
    16: '#008080', # Teer (Teal)
    17: '#DAA520', # Goldenrod
    18: '#556B2F', # Verde Oliva Escuro
    19: '#708090', # Cinza Ardósia
    20: '#C71585', # Violeta Médio
    21: '#D2691E', # Chocolate
    22: '#4682B4', # Azul Aço
    23: '#2E8B57', # Verde Mar
    24: '#FF1493'  # Rosa Profundo
}
COR_ALVO = '#FF0000' # Vermelho (Exclusivo para o Usuário Alvo)

# --- CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
@st.cache_data
def carregar_dados():
    
    #Carregando os dados do arquivo .dat
    tabela_movie = pd.read_csv('movies.dat', sep='::', engine='python')
    tabela_ratings = pd.read_csv('ratings.dat', sep='::', engine='python')
    
    #Mergiando as tabelas lidas
    tabela_merge = tabela_ratings.merge(tabela_movie, on='movieId')
    
    #Ideia de filtro para usar os filmes que as pessoas gostam (Sistema de recomendação é baseado nisso)
    tabela_bons_filmes = tabela_merge[tabela_merge['rating'] >= 3.0].copy()
    
    #Separando os generos
    tabela_bons_filmes['genres'] = tabela_bons_filmes['genres'].str.split('|')
    
    #Fazendo a nova lista de generos virar linhas -> explode
    tabela_bons_filmes = tabela_bons_filmes.explode('genres')
    
    #Contando quantos filmes bons de cada gênero o usuário viu
    tabela_contagem = tabela_bons_filmes.groupby(['userId', 'genres'])['rating'].count().unstack().fillna(0)
    
    #Transformando a contagem em PROPORÇÃO
    #Pega o total de filmes que o usuário avaliou
    total_filmes_usuario = tabela_contagem.sum(axis=1)
    
    #Divide a linha toda por esse total.
    tabela_proporcao = tabela_contagem.div(total_filmes_usuario, axis=0).fillna(0)
    
    return tabela_movie, tabela_ratings, tabela_merge, tabela_proporcao

# --- TREINAMENTO DO MODELO ---
@st.cache_resource
def treinar_modelo(tabela_proporcao, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tabela_proporcao)
    
    df_clusters = tabela_proporcao.copy()
    df_clusters['Cluster'] = clusters
    return kmeans, df_clusters

# --- FUNÇÕES DE RELATÓRIO E RECOMENDAÇÃO ---
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


#Recomenda filmes para um usuário com base no gosto do seu cluster,
#priorizando os filmes mais assistidos (populares) com as melhores notas.
def recomendar_filmes(usuario_alvo, df_clusters, df_dados_originais, df_filmes, top_n=5, min_avaliacoes=3):
    if usuario_alvo not in df_clusters.index:
        return None
    
    #Descobrir o cluster do usuário passado
    cluster_do_usuario = df_clusters.loc[usuario_alvo, 'Cluster']
    
    #Lista com os índices dos usuários(id) do mesmo cluster
    usuarios_do_cluster = df_clusters[df_clusters['Cluster'] == cluster_do_usuario].index
    
    #Lista com os filmes vistos pelo usuários indo pegar na tabela com todos os dados
    filmes_vistos_pelo_alvo = df_dados_originais[df_dados_originais['userId'] == usuario_alvo]['movieId'].unique()
    
    avaliacoes_do_cluster = df_dados_originais[df_dados_originais['userId'].isin(usuarios_do_cluster)]
    
    #Pego as avaliações dos filmes que o usuário não viu
    avaliacoes_validas = avaliacoes_do_cluster[~avaliacoes_do_cluster['movieId'].isin(filmes_vistos_pelo_alvo)]
    
    avaliacoes_unicas = avaliacoes_validas.drop_duplicates(subset=['userId', 'movieId', 'rating'])
    
    #Media e quantas pessoas avaliaram os filmes
    filmes_agrupados = avaliacoes_unicas.groupby(['movieId', 'title']).agg(
        nota_media_cluster=('rating', 'mean'),
        contagem_avaliacoes=('rating', 'count')
    ).reset_index() #volta para ser representado em colunas
    
    filmes_relevantes = filmes_agrupados[filmes_agrupados['contagem_avaliacoes'] >= min_avaliacoes]
    filmes_excelentes = filmes_relevantes[filmes_relevantes['nota_media_cluster'] >= 3.0]
    
    #Agora sim, ordenamos os excelentes pelos mais populares e pegamos o top_n(número de filmes a ser mostrado)
    top_filmes = filmes_excelentes.sort_values(
        by=['contagem_avaliacoes', 'nota_media_cluster'], 
        ascending=[False, False]
    ).head(top_n)
    
    #JoinLeft das tabelas para trazer mais informação
    top_filmes_com_generos = top_filmes.merge(df_filmes[['movieId', 'genres']], on='movieId', how='left')
    
    #Renomeando as colunas
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

# --- FUNÇÃO GERADORA DE PERSONA DO CLUSTER ---
def gerar_descricao_cluster(top_generos):
    if top_generos.empty:
        return "cluster Misterioso", "Não foi possível definir um padrão exato."
        
    genero_principal = top_generos.index[0]
    outros_generos = top_generos.index[1:].tolist()
    
    dicionario_personas = {
        'Action': ("💥 Os Adrenalinados", "Para esse cluster, o filme só é bom se tiver explosões, lutas épicas e ritmo acelerado!"),
        'Comedy': ("😂 Os Bem-Humorados", "O negócio deles é dar risada. Preferem obras leves, sarcásticas ou situações absurdas para relaxar."),
        'Drama': ("🎭 Os Profundos", "Amantes de histórias intensas, personagens complexos e tramas que trazem grandes reflexões emocionais."),
        'Horror': ("👻 Os Destemidos", "Frio na barriga é com eles mesmos! Adoram bons sustos, tensão constante e atmosferas sombrias."),
        'Romance': ("❤️ Os Apaixonados", "Histórias de amor, encontros, desencontros e finais felizes (ou trágicos) são o prato principal."),
        'Sci-Fi': ("👽 Os Visionários", "Fascinados pelo futuro, viagens no tempo e realidades alternativas. A ficção científica domina aqui."),
        'Adventure': ("🗺️ Os Exploradores", "Sempre em busca da próxima grande jornada. Adoram explorar novos mundos e culturas épicas."),
        'Fantasy': ("🧙 Os Sonhadores", "Magia, criaturas fantásticas e universos paralelos são as coisas que mais encantam esse cluster."),
        'Animation': ("🎨 Os Crianças Grandes", "Fãs de arte em movimento, curtem desde clássicos da infância até animações modernas super premiadas."),
        'Documentary': ("🧠 Os Curiosos", "Focados na realidade. Gostam de aprender e se aprofundar em fatos históricos e do mundo real."),
        'Thriller': ("🕵️ Os Investigadores", "Mistério e tensão psicológica pura. Adoram tentar adivinhar o final do filme antes de todo mundo."),
        'Crime': ("🚨 Os Detetives", "Fascinados pelo submundo, histórias de máfia, assaltos a banco e investigações policiais intensas.")
    }
    
    titulo, descricao = dicionario_personas.get(genero_principal, ("🍿 Os Cinéfilos Ecléticos", "Um cluster com gosto singular e muito interesse em boas histórias."))
    
    if outros_generos:
        descricao += f" Além de {genero_principal}, sua 'dieta de filmes' também é muito bem temperada com {', '.join(outros_generos)}."
        
    return titulo, descricao

# ==========================================
# INÍCIO DA INTERFACE (FRONT-END)
# ==========================================
st.title("🎬 Sistema de Recomendação com K-Means")

# --- MENU LATERAL (Sidebar) ---
st.sidebar.header("⚙️ Painel de Controle")
num_clusters = st.sidebar.slider("Quantidade de clusters (K)", min_value=1, max_value=25, value=5)

with st.spinner(f"Agrupando usuários em {num_clusters} clusters..."):
    movies, ratings, tabela_completa, tabela_proporcao = carregar_dados()
    modelo, df_clusters = treinar_modelo(tabela_proporcao, k=num_clusters)

lista_usuarios = df_clusters.index.tolist()
usuario_selecionado = st.sidebar.selectbox("Escolha o ID do Usuário:", lista_usuarios)
qtd_rec = st.sidebar.slider("Quantos filmes recomendar?", min_value=1, max_value=50, value=10)

cluster_atual = df_clusters.loc[usuario_selecionado, 'Cluster']

# ==========================================
# DIVISÃO EM ABAS COM MEMÓRIA DE ESTADO
# ==========================================
st.markdown("---")
aba_selecionada = st.radio(
    "Navegue pelas seções:",
    ["🍿 Recomendações", "👤 Perfil do Usuário", "👥 Detalhes do Cluster", "🌐 Todos os Clusters", "📊 Gráficos"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# ------------------------------------------------
# ABA 1: PERFIL DO USUÁRIO
# ------------------------------------------------
if aba_selecionada == "👤 Perfil do Usuário":
    st.header(f"Dados Pessoais - Usuário {usuario_selecionado}")
    st.info(f"🧠 O **K-Means** classificou este usuário no **cluster {cluster_atual}**.")
    
    total_f, media_n, generos_top, top_filmes = gerar_relatorio(usuario_selecionado, ratings, movies)
    
    col1, col2 = st.columns(2)
    col1.metric("🎬 Total de Filmes Assistidos", total_f)
    col2.metric("⭐ Média Geral de Notas", f"{media_n:.2f}")
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏆 Top Gêneros Assistidos")
        if total_f > 0:
            for genero, contagem in generos_top.head(10).items():
                porcentagem = (contagem / total_f) * 100
                st.write(f"- **{genero}**: {contagem} filmes ({porcentagem:.1f}%)")
        else:
            st.write("Nenhum dado de gênero disponível.")
            
    with col4:
        st.subheader("🍿 Filmes assistidos")
        if not top_filmes.empty:
            tabela_favoritos = top_filmes[['title', 'genres', 'rating']].rename(
                columns={'title': 'Título', 'genres': 'Gêneros', 'rating': 'Nota'}
            )
            st.dataframe(tabela_favoritos, use_container_width=True, hide_index=True)

# ------------------------------------------------
# ABA 2: DETALHES DO cluster (CLUSTER)
# ------------------------------------------------
elif aba_selecionada == "👥 Detalhes do Cluster":
    st.header(f"Visão Geral do cluster {cluster_atual}")
    st.write("Entenda o comportamento das pessoas que possuem o mesmo padrão de gosto que este usuário:")
    
    qtd_us, top_gens_c, med_filmes_c = obter_detalhes_cluster(cluster_atual, df_clusters, ratings)
    
    titulo_persona, descricao_persona = gerar_descricao_cluster(top_gens_c)
    
    st.success(f"### {titulo_persona}\n{descricao_persona}")
    st.markdown("---")
    
    c_cluster1, c_cluster2 = st.columns(2)
    c_cluster1.metric("👥 Total de Pessoas neste cluster", qtd_us)
    c_cluster2.metric("🍿 Média de Filmes Assistidos (Por pessoa)", f"{med_filmes_c:.0f} filmes")
    
    st.markdown("---")
    st.subheader("Características do cluster (Gêneros Dominantes)")
    st.write("Este cluster foi agrupada pela IA matematicamente por amar estes gêneros em comum:")
    
    for gen, prop in top_gens_c.items():
        st.write(f"- **{gen}** representa **{prop*100:.1f}%** das escolhas deste cluster.")

# ------------------------------------------------
# ABA 3: TODAS OS clusterS (NOVA VISÃO GERAL)
# ------------------------------------------------
elif aba_selecionada == "🌐 Todos os Clusters":
    st.header(f"O Mapa dos {num_clusters} clusters")
    st.write("Visão geral de como o **K-Means** dividiu todo o público do banco de dados:")
    st.markdown("---")
    
    # Passa por todos os clusters criados e gera um "card" para cada um
    for i in range(num_clusters):
        qtd_us, top_gens_c, med_filmes_c = obter_detalhes_cluster(i, df_clusters, ratings)
        titulo_persona, descricao_persona = gerar_descricao_cluster(top_gens_c)
        cor_cluster = PALETA_CLUSTERS[i]
        
        # HTML para colorir o nome do cluster combinando com o gráfico
        st.markdown(f"### <span style='color:{cor_cluster}'>cluster {i}</span>: {titulo_persona}", unsafe_allow_html=True)
        st.write(f"👥 **População:** {qtd_us} pessoas | 🎬 **Gêneros Fortes:** {', '.join(top_gens_c.index)}")
        st.info(descricao_persona)
        st.write("") # Espaçamento

# ------------------------------------------------
# ABA 4: GRÁFICOS (VISÃO 2D E 3D PADRONIZADA)
# ------------------------------------------------
elif aba_selecionada == "📊 Gráficos":
    col5, col6 = st.columns(2)
    
    with col5: 
        st.subheader("📍 Mapa de Clusters 2D (PCA)")
        st.write("Visão plana de como os clusters se dividem.")
        
        pca = PCA(n_components=2)
        dados_pca = df_clusters.drop(columns=['Cluster'])
        componentes = pca.fit_transform(dados_pca)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        cores_pontos = df_clusters['Cluster'].map(PALETA_CLUSTERS)
        
        ax.scatter(componentes[:, 0], componentes[:, 1], c=cores_pontos, alpha=0.5)
        
        idx = df_clusters.index.get_loc(usuario_selecionado)
        ax.scatter(componentes[idx, 0], componentes[idx, 1], c=COR_ALVO, s=150, edgecolors='black', marker='.', zorder=5)
        
        ax.set_xlabel('Gostos Majoritários', fontsize=10)
        ax.set_ylabel('Gostos Secundários', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        elementos_legenda = [Line2D([0], [0], marker='o', color='w', label=f'cluster {i}', 
                                    markerfacecolor=PALETA_CLUSTERS[i], markersize=8) 
                             for i in range(num_clusters)]
                             
        elementos_legenda.append(Line2D([0], [0], marker='.', color='w', label='Usuário Alvo', 
                                        markerfacecolor=COR_ALVO, markersize=15, markeredgecolor='black'))
        
        ax.legend(handles=elementos_legenda, title="clusters", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        fig.tight_layout()
        st.pyplot(fig)
    
    with col6:
        st.subheader("🪐 Mapa de Clusters 3D (Visão de Profundidade)")
        st.write("Passe o mouse pelos pontos para investigar outros usuários.")
        
        stats_usuarios = ratings.groupby('userId').agg(
            Total_Filmes=('rating', 'count'), Media=('rating', 'mean')
        ).reset_index()
        
        pca_3d = PCA(n_components=3)
        componentes_3d = pca_3d.fit_transform(dados_pca)
        
        df_3d = pd.DataFrame(componentes_3d, columns=['Eixo X (Base)', 'Eixo Y (Variação 1)', 'Eixo Z (Variação 2)'])
        df_3d['cluster (Grupo)'] = df_clusters['Cluster'].astype(str).apply(lambda x: f'cluster {x}')
        df_3d['ID do Usuário'] = df_clusters.index
        
        df_3d = df_3d.merge(stats_usuarios, left_on='ID do Usuário', right_on='userId', how='left')
        df_3d['ID do Usuário'] = df_3d['ID do Usuário'].astype(str)
        df_3d['Média de Notas'] = df_3d['Media'].round(2).astype(str)
        
        mapa_cores_plotly = {f'cluster {i}': PALETA_CLUSTERS[i] for i in range(num_clusters)}
        
        fig = px.scatter_3d(
            df_3d, x='Eixo X (Base)', y='Eixo Y (Variação 1)', z='Eixo Z (Variação 2)',
            color='cluster (Grupo)', hover_name='ID do Usuário',
            color_discrete_map=mapa_cores_plotly, 
            hover_data={ 
                'Total_Filmes': True, 'Média de Notas': True, 'cluster (Grupo)': True,
                'Eixo X (Base)': False, 'Eixo Y (Variação 1)': False, 'Eixo Z (Variação 2)': False,
                'ID do Usuário': False, 'userId': False, 'Media': False
            },
            opacity=0.6, size_max=10, width=700, height=500
        )
        
        fig.add_scatter3d(
            x=[componentes_3d[idx, 0]], y=[componentes_3d[idx, 1]], z=[componentes_3d[idx, 2]],
            mode='markers+text',
            marker=dict(color=COR_ALVO, size=12, symbol='diamond', line=dict(color='black', width=1.5)),
            name='Você', text=[f"VOCÊ (ID: {usuario_selecionado})"],
            textposition="top center", hoverinfo='skip'
        )
        
        fig.update_layout(
            legend_title_text='clusters do K-Means', margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(showbackground=False, title='Afinidade Base'),
                yaxis=dict(showbackground=False, title='Variação de Gosto'),
                zaxis=dict(showbackground=False, title='Gostos Secundários')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# ABA 5: RECOMENDAÇÕES (REATIVO)
# ------------------------------------------------
elif aba_selecionada == "🍿 Recomendações":
    st.header("🍿 Recomendações para usuário - ID: " + str(usuario_selecionado))
    st.info(f"#### Cluster {cluster_atual}")
    st.write(f"Este usuário pertence ao **cluster {cluster_atual}**. O **K-Means** analisou os gostos de pessoas semelhantes para sugerir filmes que ele ainda não viu.")
    
    st.markdown("---")
    
    with st.spinner('O **K-Means** está analisando os dados do cluster...'):
        recomendacoes = recomendar_filmes(
            usuario_selecionado, df_clusters, tabela_completa, movies, top_n=qtd_rec
        )
        
        if recomendacoes is not None and not recomendacoes.empty:
            st.dataframe(recomendacoes, use_container_width=True, hide_index=True)
            st.success("Recomendações geradas com sucesso!")
        else:
            st.warning("Não há recomendações novas suficientes para este usuário.")