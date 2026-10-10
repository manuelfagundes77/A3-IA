from fastapi import APIRouter
import pandas as pd
import os
from api.modelos import obter_caminho_ml

roteador = APIRouter()


@roteador.get("/api/metricas")
def obter_metricas():
    """Retorna as métricas de performance do modelo no teste real (2024+)."""
    caminho_csv = os.path.join(obter_caminho_ml(), 'resultados_2024.csv')

    try:
        resultados = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        return {
            "erro": "Arquivo resultados_2024.csv nao encontrado. Execute 'python treino.py' primeiro."
        }

    total_jogos = len(resultados)

    # Calcular acurácia (quem vence)
    acertos = (resultados['classe_real'] == resultados['classe_prevista']).sum()
    acuracia = round((acertos / total_jogos) * 100, 2)

    # Calcular erro médio de gols
    erro_medio = round(
        (resultados['gols_previstos'] - resultados['gols_reais']).abs().mean(), 2
    )

    return {
        "periodo_avaliado": "2024 em diante",
        "total_jogos_avaliados": total_jogos,
        "acuracia_vencedor_pct": acuracia,
        "erro_medio_gols": erro_medio,
        "descricao": "O ML-Palpites foi avaliado em jogos reais que nunca fizeram parte do treinamento."
    }

