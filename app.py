import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from data_processing import carregar_dados
from ml_models import treinar_modelo, gerar_grafico_cotovelo, gerar_grafico_silhueta
from recommender import gerar_relatorio, recomendar_filmes, obter_detalhes_cluster, gerar_descricao_cluster


# CONFIGURAÇÃO DA PÁGINA 
st.set_page_config(
    page_title="Recomendação de Filmes com K-Means", 
    layout="wide", 
    page_icon="🎬"
)

# CONFIGURAÇÃO DE CORES (25 CLUSTERS) 
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
COR_ALVO = '#FF0000' # Vermelho (usuário alvo)


# INÍCIO DA INTERFACE
st.title("🎬 Sistema de Recomendação com K-Means")

# MENU LATERAL 
st.sidebar.header("⚙️ Painel de Controle")
num_clusters = st.sidebar.slider("Quantidade de clusters (K)", min_value=1, max_value=25, value=5)

with st.spinner(f"Agrupando usuários em {num_clusters} clusters..."):
    movies, ratings, tabela_completa, tabela_proporcao = carregar_dados()
    modelo, df_clusters = treinar_modelo(tabela_proporcao, k=num_clusters)

lista_usuarios = df_clusters.index.tolist()
usuario_selecionado = st.sidebar.selectbox("Escolha o ID do Usuário:", lista_usuarios)
qtd_rec = st.sidebar.slider("Quantos filmes recomendar?", min_value=1, max_value=50, value=10)

cluster_atual = df_clusters.loc[usuario_selecionado, 'Cluster']


# DIVISÃO EM ABAS COM MEMÓRIA DE ESTADO
st.markdown("---")
aba_selecionada = st.radio(
    "Navegue pelas seções:",
    ["🍿 Recomendações", "👤 Perfil do Usuário", "👥 Detalhes do Cluster", "🌐 Todos os Clusters", "📊 Gráficos", "🗄️ Tabela de Dados"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")


# ABA 1: PERFIL DO USUÁRIO
if aba_selecionada == "👤 Perfil do Usuário":
    st.header(f"Dados Pessoais - Usuário {usuario_selecionado}")
    st.info(f"🧠 O **K-Means** classificou este usuário no **cluster {cluster_atual}**.")
    
    total_f, media_n, generos_top, top_filmes = gerar_relatorio(usuario_selecionado, ratings, movies)
    
    col1, col2 = st.columns(2)
    col1.metric("🎬 Total de Filmes", total_f)
    col2.metric("⭐ Média Geral de Notas", f"{media_n:.2f}")
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏆 Top Gêneros")
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


# ABA 2: DETALHES DO CLUSTER
elif aba_selecionada == "👥 Detalhes do Cluster":
    st.header(f"Visão Geral do cluster {cluster_atual}")
    st.write("Entenda o comportamento das pessoas que possuem o mesmo padrão de gosto que este usuário:")
    
    qtd_us, top_gens_c, med_filmes_c = obter_detalhes_cluster(cluster_atual, df_clusters, ratings)
    
    titulo_persona, descricao_persona = gerar_descricao_cluster(top_gens_c)
    
    # Tratamento visual especial caso seja o grupo dos insatisfeitos
    if top_gens_c.sum() == 0:
        st.error(f"### {titulo_persona}\n{descricao_persona}")
    else:
        st.success(f"### {titulo_persona}\n{descricao_persona}")
        
    st.markdown("---")
    
    c_cluster1, c_cluster2 = st.columns(2)
    c_cluster1.metric("👥 Total de Pessoas neste cluster", qtd_us)
    c_cluster2.metric("🍿 Média de Filmes Assistidos (Por pessoa)", f"{med_filmes_c:.0f} filmes")
    
    st.markdown("---")
    st.subheader("Características do cluster (Gêneros Dominantes)")
    st.write("Este cluster foi agrupada pela IA matematicamente por amar estes gêneros em comum:")
    
    # Trava para não imprimir lista de 0%
    if top_gens_c.sum() == 0:
        st.warning("Este cluster não possui gêneros dominantes, pois é formado inteiramente por usuários que não deram notas positivas.")
    else:
        for gen, prop in top_gens_c.items():
            if prop > 0:
                st.write(f"- **{gen}** representa **{prop*100:.1f}%** das escolhas deste cluster.")


# ABA 3: TODAS OS CLUSTERS (NOVA VISÃO GERAL)
elif aba_selecionada == "🌐 Todos os Clusters":
    st.header(f"Visão Geral: {num_clusters} Clusters")
    st.write("Resumo do agrupamento gerado pelo modelo K-Means para toda a base de usuários.")
    st.markdown("---")
    
    # Passa por todos os clusters criados e gera um item de lista limpo para cada um
    for i in range(num_clusters):
        qtd_us, top_gens_c, med_filmes_c = obter_detalhes_cluster(i, df_clusters, ratings)
        titulo_persona, descricao_persona = gerar_descricao_cluster(top_gens_c)
        
        st.subheader(f"Cluster {i}: {titulo_persona}")
        
        # Verifica se é o cluster dos insatisfeitos para adaptar o texto
        if top_gens_c.sum() == 0:
            st.markdown(f"**População:** {qtd_us} usuários | **Gêneros Dominantes:** Nenhum")
        else:
            generos_validos = top_gens_c[top_gens_c > 0].index
            generos_str = ', '.join(generos_validos)
            st.markdown(f"**População:** {qtd_us} usuários | **Gêneros Dominantes:** {generos_str}")
            
        st.write(descricao_persona)
        
        st.markdown("---")


# ABA 4: GRÁFICOS
elif aba_selecionada == "📊 Gráficos":
    
    st.header("📊 Visualização dos Clusters")
    st.subheader("📍 Mapa de Clusters 2D (PCA)")
    st.write("Visão plana de como os clusters se dividem.")
        
    pca = PCA(n_components=2)
    dados_pca = df_clusters.drop(columns=['Cluster'])
    componentes = pca.fit_transform(dados_pca)
    
    fig, ax = plt.subplots(figsize=(5, 3))
        
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
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")
    
    st.header("⚙️ Análise Técnica: Otimização do Modelo")
    st.write("Abaixo, analisamos a compactação dos grupos e a separação entre eles.")

    col1, col2 = st.columns(2)
    
    with col1:
        
        # Exibindo o Método do Cotovelo
        st.subheader("1. Método do Cotovelo (Inércia)")
        fig_cotovelo = gerar_grafico_cotovelo(tabela_proporcao)
        st.pyplot(fig_cotovelo)
        st.info("💡 Buscamos o ponto onde a queda da curva suaviza significativamente.")

    with col2:
        
        # Exibindo o Método da Silhueta
        st.subheader("2. Análise da Silhueta")
        fig_silhueta = gerar_grafico_silhueta(tabela_proporcao)
        st.pyplot(fig_silhueta)
        st.success("💡 Quanto mais próximo de 1.0, melhor a definição e separação dos clusters.")

# ABA 5: RECOMENDAÇÕES (REATIVO)
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
            

# ABA 6: TABELA DE DADOS E BASTIDORES
elif aba_selecionada == "🗄️ Tabela de Dados":
    st.header("🗄️ Repositório de Dados")
    
    # Criando sub-abas internas
    sub_dados1, sub_dados2 = st.tabs(["📊 Matriz do Modelo (Proporções)", "🎞️ Dados Brutos (Merge)"])

    with sub_dados1:
        st.subheader("Matriz de Entrada do K-Means")
        st.write("Cada linha representa um usuário e sua afinidade percentual por gênero.")
        
        # Filtro de Cluster
        if st.checkbox("Filtrar apenas usuários da mesma tribo", key="filtro_cluster"):
            tabela_exibicao = df_clusters[df_clusters['Cluster'] == cluster_atual]
            st.success(f"Exibindo {len(tabela_exibicao)} usuários do Cluster {cluster_atual}")
        else:
            tabela_exibicao = df_clusters

        st.dataframe(tabela_exibicao, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Distribuição por Grupo")
        contagem = df_clusters['Cluster'].value_counts().sort_index().reset_index()
        contagem.columns = ['Cluster', 'Quantidade de Usuários']
        st.dataframe(contagem, hide_index=True)

    with sub_dados2:
        st.subheader("Histórico Bruto de Avaliações")
        st.write("Esta é a união das tabelas de Filmes e Notas antes do tratamento estatístico.")
        
        # Filtro por usuário alvo
        if st.checkbox("Mostrar apenas avaliações do Usuário Selecionado", value=True):
            dados_brutos = tabela_completa[tabela_completa['userId'] == usuario_selecionado]
        else:
            # Mostra uma amostra se a tabela for muito grande para não travar o navegador
            dados_brutos = tabela_completa.head(1000)
            st.warning("Exibindo as primeiras 1000 linhas por performance.")

        # Formatando a exibição
        st.dataframe(
            dados_brutos[['userId', 'movieId', 'title', 'genres', 'rating']], 
            use_container_width=True, 
            hide_index=True
        )