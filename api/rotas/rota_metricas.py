from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from api.modelos import obter_caminho_resultados, VERSAO_PADRAO

roteador = APIRouter()


@roteador.get("/api/metricas")
def obter_metricas(versao: str = VERSAO_PADRAO):
    """Retorna as métricas de performance do modelo no teste real (2024+)."""
    caminho_csv = obter_caminho_resultados(versao)

    try:
        resultados = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Resultados da versao '{versao}' nao encontrados ({os.path.basename(caminho_csv)}). Execute 'python treino.py' primeiro."
        )

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

