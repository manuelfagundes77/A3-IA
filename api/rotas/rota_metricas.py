from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from api.modelos import obter_caminho_resultados, VERSAO_PADRAO

roteador = APIRouter()

# Dicionários descritivos de features por versão para atender ao /model-info
FEATURES_V1 = [
    {"nome": "home_form_gf", "descricao": "Média de gols feitos pelo mandante nos últimos 5 jogos"},
    {"nome": "home_form_ga", "descricao": "Média de gols sofridos pelo mandante nos últimos 5 jogos"},
    {"nome": "home_form_pts", "descricao": "Média de pontos conquistados pelo mandante nos últimos 5 jogos"},
    {"nome": "away_form_gf", "descricao": "Média de gols feitos pelo visitante nos últimos 5 jogos"},
    {"nome": "away_form_ga", "descricao": "Média de gols sofridos pelo visitante nos últimos 5 jogos"},
    {"nome": "away_form_pts", "descricao": "Média de pontos conquistados pelo visitante nos últimos 5 jogos"},
    {"nome": "h2h_home_win_pct", "descricao": "Taxa de vitórias do mandante no confronto direto recente (H2H)"},
    {"nome": "h2h_away_win_pct", "descricao": "Taxa de vitórias do visitante no confronto direto recente (H2H)"},
    {"nome": "h2h_draw_pct", "descricao": "Taxa de empates no confronto direto recente (H2H)"},
    {"nome": "home_hist_gf_casa", "descricao": "Média histórica de gols feitos pelo mandante jogando em casa"},
    {"nome": "home_hist_ga_casa", "descricao": "Média histórica de gols sofridos pelo mandante jogando em casa"},
    {"nome": "away_hist_gf_fora", "descricao": "Média histórica de gols feitos pelo visitante jogando fora"},
    {"nome": "away_hist_ga_fora", "descricao": "Média histórica de gols sofridos pelo visitante jogando fora"}
]

FEATURES_V2 = FEATURES_V1 + [
    {"nome": "prob_implicita_mandante", "descricao": "Probabilidade implícita da odd para vitória do mandante"},
    {"nome": "prob_implicita_empate", "descricao": "Probabilidade implícita da odd para empate"},
    {"nome": "prob_implicita_visitante", "descricao": "Probabilidade implícita da odd para vitória do visitante"},
    {"nome": "prob_norm_mandante", "descricao": "Probabilidade do mandante normalizada sem a margem da casa (Vig)"},
    {"nome": "prob_norm_empate", "descricao": "Probabilidade do empate normalizada sem a margem da casa (Vig)"},
    {"nome": "prob_norm_visitante", "descricao": "Probabilidade do visitante normalizada sem a margem da casa (Vig)"},
    {"nome": "razao_odds", "descricao": "Razão de odds (Odd Visitante / Odd Mandante)"},
    {"nome": "diff_pts", "descricao": "Diferença direta de momento em pontos (Mandante - Visitante)"},
    {"nome": "diff_saldo", "descricao": "Diferença direta de saldo de gols recente"},
    {"nome": "diff_ataque_defesa", "descricao": "Diferencial de força ofensiva do mandante contra defensiva do visitante"},
    {"nome": "diff_h2h", "descricao": "Diferença de histórico no confronto direto"},
    {"nome": "diff_mando", "descricao": "Diferencial de força histórica de mando de campo"}
]


@roteador.get("/api/metricas")
@roteador.get("/model-info")
def obter_metricas(versao: str = VERSAO_PADRAO):
    """
    Retorna as métricas de performance do modelo no teste real (2024+),
    o tipo de algoritmo e a lista completa de features usadas.
    Atende tanto a rota original /api/metricas quanto o endpoint obrigatório /model-info.
    """
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
    acuracia = float(round((acertos / total_jogos) * 100, 2))

    # Calcular erro médio de gols
    erro_medio = float(round(
        (resultados['gols_previstos'] - resultados['gols_reais']).abs().mean(), 2
    ))

    # Acurácia de treino (aprendizado 2012-2023) por versão
    acuracia_treino = 69.40 if versao == "v2" else 57.07
    ganho_sobre_sorte = float(round(acuracia - 33.33, 2))

    # Metadados do modelo
    if versao == "v2":
        tipo_modelo = "XGBoost (Extreme Gradient Boosting Classifier & Regressor)"
        algoritmo_classificador = "XGBClassifier (1000 estimadores, max_depth=6, lr=0.01)"
        algoritmo_regressor = "XGBRegressor (1000 estimadores, max_depth=5, lr=0.01)"
        features_lista = FEATURES_V2
    else:
        tipo_modelo = "Random Forest (Scikit-Learn Classifier & Regressor)"
        algoritmo_classificador = "RandomForestClassifier (200 árvores, max_depth=7)"
        algoritmo_regressor = "RandomForestRegressor (200 árvores, max_depth=7)"
        features_lista = FEATURES_V1

    return {
        "versao": versao,
        "tipo_modelo": tipo_modelo,
        "algoritmo_classificador": algoritmo_classificador,
        "algoritmo_regressor": algoritmo_regressor,
        "total_features": len(features_lista),
        "features_usadas": features_lista,
        "periodo_treino": "2012 a 2023",
        "periodo_avaliado": "2024 em diante",
        "total_jogos_avaliados": total_jogos,
        "acuracia_treino_pct": acuracia_treino,
        "acuracia_vencedor_pct": acuracia,
        "baseline_aleatorio_pct": 33.33,
        "ganho_vs_aleatorio_pct": ganho_sobre_sorte,
        "erro_medio_gols": erro_medio,
        "descricao": "O ML-Palpites foi avaliado em jogos reais que nunca fizeram parte do treinamento."
    }
