import os
import pickle
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

roteador = APIRouter()

# Caminho do modelo salvo de clusters
CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_KMEANS = os.path.join(CAMINHO_BASE, '..', 'ml', 'kmeans_times.pkl')

dados_cluster_cache = None


def carregar_dados_cluster():
    global dados_cluster_cache
    if dados_cluster_cache is None:
        if os.path.exists(ARQUIVO_KMEANS):
            with open(ARQUIVO_KMEANS, 'rb') as f:
                dados_cluster_cache = pickle.load(f)
        else:
            dados_cluster_cache = {
                'times': {},
                'clusters': {}
            }
    return dados_cluster_cache


class PedidoCluster(BaseModel):
    time_casa: Optional[str] = "Palmeiras"
    time_fora: Optional[str] = "Flamengo RJ"
    versao: Optional[str] = "v1"


def processar_resposta_cluster(time_casa: str, time_fora: str):
    dados = carregar_dados_cluster()
    dic_times = dados.get('times', {})
    dic_clusters = dados.get('clusters', {})

    padrao_casa = {
        'time': time_casa,
        'cluster': 1,
        'perfil': 'Competitivo / Série A',
        'media_pontos': 1.40,
        'media_gols_pro': 1.25,
        'media_gols_contra': 1.15,
        'aprov_casa': 1.80,
        'aprov_fora': 1.00
    }

    padrao_fora = {
        'time': time_fora,
        'cluster': 1,
        'perfil': 'Competitivo / Série A',
        'media_pontos': 1.40,
        'media_gols_pro': 1.25,
        'media_gols_contra': 1.15,
        'aprov_casa': 1.80,
        'aprov_fora': 1.00
    }

    info_casa = dic_times.get(time_casa, padrao_casa)
    info_fora = dic_times.get(time_fora, padrao_fora)

    mesmo_cluster = (info_casa['cluster'] == info_fora['cluster'])
    if mesmo_cluster:
        relacao = f"Duelo equilibrado entre dois clubes do mesmo escalão ({info_casa['perfil']})."
    elif info_casa['cluster'] < info_fora['cluster']:
        relacao = f"Mandante ({info_casa['time']}) pertence a um escalão superior ({info_casa['perfil']}) em relação ao visitante ({info_fora['perfil']})."
    else:
        relacao = f"Visitante ({info_fora['time']}) pertence a um escalão superior ({info_fora['perfil']}), desafiando a força de mando do {info_casa['time']}."

    return {
        "confronto": f"{time_casa} vs {time_fora}",
        "mandante": info_casa,
        "visitante": info_fora,
        "mesmo_cluster": mesmo_cluster,
        "relacao_confronto": relacao,
        "total_clusters": len(dic_clusters),
        "clusters_gerais": dic_clusters,
        "grafico_disponivel": "grafico_clusters_kmeans.png",
        "descricao": "Segmentação não-supervisionada gerada por algoritmo K-Means nos dados agregados do Brasileirão."
    }


@roteador.post("/cluster")
def obter_cluster_post(pedido: PedidoCluster):
    """
    Endpoint obrigatório /cluster (POST):
    Retorna a segmentação K-Means dos dois times selecionados e o catálogo geral de perfis.
    """
    time_casa = pedido.time_casa or "Palmeiras"
    time_fora = pedido.time_fora or "Flamengo RJ"
    return processar_resposta_cluster(time_casa, time_fora)


@roteador.get("/cluster")
def obter_cluster_get(time_casa: str = "Palmeiras", time_fora: str = "Flamengo RJ", versao: str = "v1"):
    """
    Endpoint obrigatório /cluster (GET):
    Permite abrir diretamente no navegador para visualizar o JSON puro da segmentação K-Means.
    """
    return processar_resposta_cluster(time_casa, time_fora)

