import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@st.cache_resource
def treinar_modelo(tabela_proporcao, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tabela_proporcao)
    
    df_clusters = tabela_proporcao.copy()
    df_clusters['Cluster'] = clusters
    return kmeans, df_clusters


@st.cache_resource
def gerar_grafico_cotovelo(tabela_proporcao):
    inercia = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(tabela_proporcao)
        inercia.append(km.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inercia, marker='o', color='#1f77b4', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax.set_ylabel('Inércia (WCSS)', fontsize=12)
    ax.set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

@st.cache_resource
def gerar_grafico_silhueta(tabela_proporcao):
    silhueta = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(tabela_proporcao)
        silhueta.append(silhouette_score(tabela_proporcao, clusters))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, silhueta, marker='s', color='#2ca02c', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax.set_ylabel('Score da Silhueta', fontsize=12)
    ax.set_title('Análise da Silhueta', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig
