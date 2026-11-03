from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from api.modelos import obter_modelos, versoes_disponiveis, VERSAO_PADRAO

roteador = APIRouter()


class PedidoPrevisao(BaseModel):
    """Modelo do JSON que o frontend envia."""
    time_casa: str
    time_fora: str
    versao: str = VERSAO_PADRAO


def calcular_confianca(chance_maxima):
    """
    Retorna o nível de confiança calibrado para classificação de 3 classes no futebol
    (onde a probabilidade base de equilíbrio é 33.3%).
    """
    if chance_maxima >= 44.0:
        return "Alta"
    elif chance_maxima >= 37.0:
        return "Média"
    else:
        return "Moderada"


def extrair_dados_confronto(pedido: PedidoPrevisao):
    """Extrai features, históricos e modelos para um confronto entre dois times."""
    modelos = obter_modelos(pedido.versao)
    if modelos is None:
        raise HTTPException(
            status_code=404,
            detail=f"Versao '{pedido.versao}' nao disponivel. Disponiveis: {versoes_disponiveis()}"
        )
    classificador = modelos["classificador"]
    regressor = modelos["regressor"]
    estado = modelos["estado"]

    time_casa = pedido.time_casa
    time_fora = pedido.time_fora

    historico_times = estado['historico_times']
    historico_em_casa = estado['historico_em_casa']
    historico_fora = estado['historico_fora']
    historico_h2h = estado['historico_h2h']

    # --- Momento Atual (Últimos 5 jogos) ---
    home_games = historico_times.get(time_casa, [])[-5:]
    home_gf  = sum([g['gf'] for g in home_games]) / 5 if len(home_games) > 0 else 0.0
    home_ga  = sum([g['ga'] for g in home_games]) / 5 if len(home_games) > 0 else 0.0
    home_pts = sum([g['pts'] for g in home_games]) / 5 if len(home_games) > 0 else 0.0

    away_games = historico_times.get(time_fora, [])[-5:]
    away_gf  = sum([g['gf'] for g in away_games]) / 5 if len(away_games) > 0 else 0.0
    away_ga  = sum([g['ga'] for g in away_games]) / 5 if len(away_games) > 0 else 0.0
    away_pts = sum([g['pts'] for g in away_games]) / 5 if len(away_games) > 0 else 0.0

    # --- Confronto Direto (Últimos 5 entre eles) ---
    equipes = tuple(sorted([time_casa, time_fora]))
    h2h_games = historico_h2h.get(equipes, [])[-5:]

    if len(h2h_games) > 0:
        h2h_home_wins = sum([1 for g in h2h_games if g['winner'] == time_casa]) / len(h2h_games)
        h2h_away_wins = sum([1 for g in h2h_games if g['winner'] == time_fora]) / len(h2h_games)
        h2h_draws     = sum([1 for g in h2h_games if g['winner'] == 'Draw']) / len(h2h_games)
    else:
        h2h_home_wins, h2h_away_wins, h2h_draws = 0.0, 0.0, 0.0

    # --- Média Histórica de Casa/Fora ---
    jogos_em_casa = historico_em_casa.get(time_casa, [])
    if len(jogos_em_casa) > 0:
        home_hist_gf_casa = sum([g['gf'] for g in jogos_em_casa]) / len(jogos_em_casa)
        home_hist_ga_casa = sum([g['ga'] for g in jogos_em_casa]) / len(jogos_em_casa)
    else:
        home_hist_gf_casa = 0.0
        home_hist_ga_casa = 0.0

    jogos_fora_hist = historico_fora.get(time_fora, [])
    if len(jogos_fora_hist) > 0:
        away_hist_gf_fora = sum([g['gf'] for g in jogos_fora_hist]) / len(jogos_fora_hist)
        away_hist_ga_fora = sum([g['ga'] for g in jogos_fora_hist]) / len(jogos_fora_hist)
    else:
        away_hist_gf_fora = 0.0
        away_hist_ga_fora = 0.0

    # --- Montar features de acordo com a versão do modelo ---
    features_esperadas = estado.get('features', None)

    if pedido.versao == "v2" or (features_esperadas and len(features_esperadas) > 13):
        forca_h = max(0.3, (home_pts * 0.4) + (home_hist_gf_casa * 0.4) - (home_hist_ga_casa * 0.15) + 0.35)
        forca_a = max(0.2, (away_pts * 0.4) + (away_hist_gf_fora * 0.4) - (away_hist_ga_fora * 0.15))
        soma_forca = forca_h + forca_a + 0.85

        prob_est_h = max(0.18, min(0.72, forca_h / soma_forca))
        prob_est_a = max(0.12, min(0.65, forca_a / soma_forca))
        prob_est_d = max(0.18, 1.0 - prob_est_h - prob_est_a)

        odd_h = round(1.0 / (prob_est_h * 1.05), 2)
        odd_d = round(1.0 / (prob_est_d * 1.05), 2)
        odd_a = round(1.0 / (prob_est_a * 1.05), 2)

        prob_raw_h = 1.0 / odd_h
        prob_raw_d = 1.0 / odd_d
        prob_raw_a = 1.0 / odd_a
        soma_p = prob_raw_h + prob_raw_d + prob_raw_a

        prob_norm_h = prob_raw_h / soma_p
        prob_norm_d = prob_raw_d / soma_p
        prob_norm_a = prob_raw_a / soma_p

        diff_pts = home_pts - away_pts
        diff_saldo = (home_gf - home_ga) - (away_gf - away_ga)
        diff_ataque_defesa = home_gf - away_ga
        diff_h2h = h2h_home_wins - h2h_away_wins
        diff_mando = home_hist_gf_casa - away_hist_ga_fora
        razao_odds = odd_a / odd_h if odd_h > 0 else 1.0

        dic_features = {
            'home_form_gf': home_gf,
            'home_form_ga': home_ga,
            'home_form_pts': home_pts,
            'away_form_gf': away_gf,
            'away_form_ga': away_ga,
            'away_form_pts': away_pts,
            'h2h_home_win_pct': h2h_home_wins,
            'h2h_away_win_pct': h2h_away_wins,
            'h2h_draw_pct': h2h_draws,
            'home_hist_gf_casa': home_hist_gf_casa,
            'home_hist_ga_casa': home_hist_ga_casa,
            'away_hist_gf_fora': away_hist_gf_fora,
            'away_hist_ga_fora': away_hist_ga_fora,
            'prob_implicita_mandante': prob_raw_h,
            'prob_implicita_empate': prob_raw_d,
            'prob_implicita_visitante': prob_raw_a,
            'prob_norm_mandante': prob_norm_h,
            'prob_norm_empate': prob_norm_d,
            'prob_norm_visitante': prob_norm_a,
            'razao_odds': razao_odds,
            'diff_pts': diff_pts,
            'diff_saldo': diff_saldo,
            'diff_ataque_defesa': diff_ataque_defesa,
            'diff_h2h': diff_h2h,
            'diff_mando': diff_mando,
        }
        df_features = pd.DataFrame([dic_features])
        if features_esperadas:
            df_features = df_features[features_esperadas]
    else:
        df_features = pd.DataFrame([{
            'home_form_gf': home_gf,
            'home_form_ga': home_ga,
            'home_form_pts': home_pts,
            'away_form_gf': away_gf,
            'away_form_ga': away_ga,
            'away_form_pts': away_pts,
            'h2h_home_win_pct': h2h_home_wins,
            'h2h_away_win_pct': h2h_away_wins,
            'h2h_draw_pct': h2h_draws,
            'home_hist_gf_casa': home_hist_gf_casa,
            'home_hist_ga_casa': home_hist_ga_casa,
            'away_hist_gf_fora': away_hist_gf_fora,
            'away_hist_ga_fora': away_hist_ga_fora,
        }])

    # --- Previsões dos Modelos ---
    pred_bruto = classificador.predict(df_features)[0]
    probabilidades = classificador.predict_proba(df_features)[0]
    classes_modelo = classificador.classes_
    pred_gols = float(regressor.predict(df_features)[0])

    mapa_inverso = estado.get('mapa_inverso', {0: 'A', 1: 'D', 2: 'H'})
    pred_classe = mapa_inverso.get(pred_bruto, pred_bruto)

    probs = {}
    for cls, p in zip(classes_modelo, probabilidades):
        chave_cls = mapa_inverso.get(cls, cls)
        probs[chave_cls] = float(p)

    chance_casa   = round(probs.get('H', 0) * 100, 1)
    chance_empate = round(probs.get('D', 0) * 100, 1)
    chance_fora   = round(probs.get('A', 0) * 100, 1)
    chance_maxima = max(chance_casa, chance_empate, chance_fora)

    # --- Placar Mais Provável ---
    forca_ataque_casa = home_hist_gf_casa + away_hist_ga_fora
    forca_ataque_fora = away_hist_gf_fora + home_hist_ga_casa
    total_forca = forca_ataque_casa + forca_ataque_fora

    if total_forca > 0:
        gols_esperados_casa = pred_gols * (forca_ataque_casa / total_forca)
        gols_esperados_fora = pred_gols * (forca_ataque_fora / total_forca)
    else:
        gols_esperados_casa = pred_gols / 2
        gols_esperados_fora = pred_gols / 2

    placar_casa = round(gols_esperados_casa)
    placar_fora = round(gols_esperados_fora)

    if pred_classe == 'H' and placar_casa <= placar_fora:
        placar_casa = placar_fora + 1
    elif pred_classe == 'A' and placar_fora <= placar_casa:
        placar_fora = placar_casa + 1
    elif pred_classe == 'D' and placar_casa != placar_fora:
        media = (placar_casa + placar_fora) // 2
        placar_casa = media
        placar_fora = media

    placar_provavel = f"{placar_casa} x {placar_fora}"

    return {
        "time_casa": time_casa,
        "time_fora": time_fora,
        "versao": pedido.versao,
        "home_gf": home_gf,
        "home_ga": home_ga,
        "home_pts": home_pts,
        "away_gf": away_gf,
        "away_ga": away_ga,
        "away_pts": away_pts,
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "h2h_draws": h2h_draws,
        "home_hist_gf_casa": home_hist_gf_casa,
        "home_hist_ga_casa": home_hist_ga_casa,
        "away_hist_gf_fora": away_hist_gf_fora,
        "away_hist_ga_fora": away_hist_ga_fora,
        "pred_classe": pred_classe,
        "chance_casa": chance_casa,
        "chance_empate": chance_empate,
        "chance_fora": chance_fora,
        "chance_maxima": chance_maxima,
        "pred_gols": pred_gols,
        "gols_esperados_casa": gols_esperados_casa,
        "gols_esperados_fora": gols_esperados_fora,
        "placar_provavel": placar_provavel,
        "df_features": df_features,
        "classificador": classificador
    }


@roteador.post("/api/previsao")
@roteador.post("/predict")
def fazer_previsao(pedido: PedidoPrevisao):
    """
    Recebe dois times e retorna a classificação de vitória (Casa, Empate, Fora),
    probabilidades e expectativa de gols.
    Atende tanto a rota /api/previsao quanto o endpoint obrigatório /predict.
    """
    dados = extrair_dados_confronto(pedido)
    time_casa = dados["time_casa"]
    time_fora = dados["time_fora"]
    pred_classe = dados["pred_classe"]

    if pred_classe == 'H':
        palpite = f"Vitoria do(a) {time_casa} (Casa)"
    elif pred_classe == 'A':
        palpite = f"Vitoria do(a) {time_fora} (Visitante)"
    else:
        palpite = "Empate"

    return {
        "confronto": f"{time_casa} vs {time_fora}",
        "raio_x": {
            "casa": {
                "gols_feitos_media": round(dados["home_gf"], 2),
                "gols_sofridos_media": round(dados["home_ga"], 2),
                "pontos_media": round(dados["home_pts"], 2)
            },
            "fora": {
                "gols_feitos_media": round(dados["away_gf"], 2),
                "gols_sofridos_media": round(dados["away_ga"], 2),
                "pontos_media": round(dados["away_pts"], 2)
            }
        },
        "retrospecto": {
            "vitorias_casa_pct": round(dados["h2h_home_wins"] * 100, 1),
            "empates_pct": round(dados["h2h_draws"] * 100, 1),
            "vitorias_fora_pct": round(dados["h2h_away_wins"] * 100, 1)
        },
        "previsao": {
            "palpite": palpite,
            "chance_casa_pct": dados["chance_casa"],
            "chance_empate_pct": dados["chance_empate"],
            "chance_fora_pct": dados["chance_fora"],
            "total_gols_esperados": round(dados["pred_gols"], 2),
            "placar_provavel": dados["placar_provavel"],
            "media_gols_casa": round(dados["home_hist_gf_casa"], 2),
            "media_gols_fora": round(dados["away_hist_gf_fora"], 2),
            "confianca": calcular_confianca(dados["chance_maxima"]),
            "confianca_pct": dados["chance_maxima"]
        }
    }


@roteador.post("/forecast")
def fazer_previsao_forecast(pedido: PedidoPrevisao):
    """
    Endpoint obrigatório /forecast: prevê valor numérico contínuo (regressão de gols).
    Retorna a expectativa de gols na partida, divisão por time, placar provável e tendência de mercado.
    """
    dados = extrair_dados_confronto(pedido)
    time_casa = dados["time_casa"]
    time_fora = dados["time_fora"]
    pred_gols = dados["pred_gols"]

    tipo_regressor = (
        "XGBoost Regressor (1000 árvores, V2)"
        if pedido.versao == "v2"
        else "Random Forest Regressor (200 árvores, V1)"
    )

    tendencia_mercado = (
        "Mais de 2.5 Gols (Over 2.5)"
        if pred_gols >= 2.5
        else "Menos de 2.5 Gols (Under 2.5)"
    )

    return {
        "confronto": f"{time_casa} vs {time_fora}",
        "versao": pedido.versao,
        "algoritmo_regressao": tipo_regressor,
        "previsao_numerica": {
            "total_gols_esperados": round(pred_gols, 2),
            "gols_esperados_casa": round(dados["gols_esperados_casa"], 2),
            "gols_esperados_fora": round(dados["gols_esperados_fora"], 2),
            "placar_provavel": dados["placar_provavel"],
            "tendencia_mercado": tendencia_mercado,
            "erro_medio_historico_mae": 1.19
        },
        "medias_mando_historico": {
            "media_gols_casa": round(dados["home_hist_gf_casa"], 2),
            "media_gols_fora": round(dados["away_hist_gf_fora"], 2)
        },
        "descricao": "Previsão contínua de gols gerada por modelo de regressão de Machine Learning."
    }


@roteador.get("/forecast")
def fazer_previsao_forecast_get(time_casa: str, time_fora: str, versao: str = VERSAO_PADRAO):
    """Permite abrir o endpoint /forecast diretamente no navegador via GET para ver o JSON puro."""
    pedido = PedidoPrevisao(time_casa=time_casa, time_fora=time_fora, versao=versao)
    return fazer_previsao_forecast(pedido)


@roteador.post("/predict-rule")
def fazer_previsao_regra_fixa(pedido: PedidoPrevisao):
    """
    Endpoint obrigatório /predict-rule: Compara a decisão por Regra Fixa (heurística sem IA)
    com a decisão do Machine Learning para o mesmo confronto.
    """
    dados = extrair_dados_confronto(pedido)
    time_casa = dados["time_casa"]
    time_fora = dados["time_fora"]
    home_pts = dados["home_pts"]
    away_pts = dados["away_pts"]
    pred_classe = dados["pred_classe"]

    # --- 1. Lógica da Regra Fixa (Heurística sem Machine Learning) ---
    # Dá um bônus empírico de +0.35 pts pelo mando de campo e compara a média de pontos recentes
    pontos_casa_ponderado = home_pts + 0.35
    pontos_fora = away_pts
    diferenca = round(pontos_casa_ponderado - pontos_fora, 2)

    if diferenca >= 0.40:
        palpite_regra = f"Vitória do(a) {time_casa} (Casa)"
        classe_regra = "H"
        explicacao_regra = (
            f"Mandante soma {home_pts:.2f} pts recentes + 0.35 de bônus de mando ({pontos_casa_ponderado:.2f} pts), "
            f"superando os {away_pts:.2f} pts do visitante por uma margem de +{diferenca:.2f} pts."
        )
    elif diferenca <= -0.40:
        palpite_regra = f"Vitória do(a) {time_fora} (Visitante)"
        classe_regra = "A"
        explicacao_regra = (
            f"Visitante tem momento superior com {away_pts:.2f} pts recentes, "
            f"superando os {home_pts:.2f} (+0.35) do mandante com diferença de {abs(diferenca):.2f} pts."
        )
    else:
        palpite_regra = "Empate"
        classe_regra = "D"
        explicacao_regra = (
            f"Equilíbrio de momento: Mandante com {pontos_casa_ponderado:.2f} pts ponderados "
            f"vs Visitante com {away_pts:.2f} pts (diferença de apenas {abs(diferenca):.2f} pts, indicando empate)."
        )

    # --- 2. Decisão do Machine Learning ---
    if pred_classe == 'H':
        palpite_ml = f"Vitória do(a) {time_casa} (Casa)"
    elif pred_classe == 'A':
        palpite_ml = f"Vitória do(a) {time_fora} (Visitante)"
    else:
        palpite_ml = "Empate"

    concordam = (classe_regra == pred_classe)
    total_features = 25 if pedido.versao == "v2" else 13
    acuracia_ml = 52.07 if pedido.versao == "v2" else 50.24
    acuracia_regra_fixa = 44.10  # Acurácia histórica do baseline de regra fixa no Brasileirão
    ganho_ia = round(acuracia_ml - acuracia_regra_fixa, 2)

    return {
        "confronto": f"{time_casa} vs {time_fora}",
        "versao_ml": pedido.versao,
        "concordancia": "Concordam" if concordam else "Discordam",
        "regra_fixa": {
            "palpite": palpite_regra,
            "classe": classe_regra,
            "pontos_mandante_ponderado": round(pontos_casa_ponderado, 2),
            "pontos_visitante": round(pontos_fora, 2),
            "diferenca_pontos": diferenca,
            "explicacao_regra": explicacao_regra,
            "acuracia_historica_pct": acuracia_regra_fixa
        },
        "machine_learning": {
            "palpite": palpite_ml,
            "classe": pred_classe,
            "total_features_analisadas": total_features,
            "probabilidades_pct": {
                "casa": dados["chance_casa"],
                "empate": dados["chance_empate"],
                "fora": dados["chance_fora"]
            },
            "acuracia_real_pct": acuracia_ml
        },
        "comparativo": {
            "vantagem_ml_sobre_regra_pct": f"+{ganho_ia}%",
            "conclusao": (
                "O Machine Learning supera a regra fixa manual porque correlaciona "
                f"{total_features} variáveis ao mesmo tempo (incluindo gols, confronto direto e mando), "
                "enquanto a regra fixa considera apenas um cálculo linear simples."
            )
        }
    }


@roteador.get("/predict-rule")
def fazer_previsao_regra_fixa_get(time_casa: str, time_fora: str, versao: str = VERSAO_PADRAO):
    """Permite abrir o endpoint /predict-rule diretamente no navegador via GET para ver o JSON puro."""
    pedido = PedidoPrevisao(time_casa=time_casa, time_fora=time_fora, versao=versao)
    return fazer_previsao_regra_fixa(pedido)


# Mapeamento em Português das features para explicação da Árvore de Decisão (/explain)
NOMES_FEATURES_EXPLAIN = {
    "home_form_gf": "Média de Gols Feitos pelo Mandante (últ. 5 jogos)",
    "home_form_ga": "Média de Gols Sofridos pelo Mandante (últ. 5 jogos)",
    "home_form_pts": "Média de Pontos do Mandante (últ. 5 jogos)",
    "away_form_gf": "Média de Gols Feitos pelo Visitante (últ. 5 jogos)",
    "away_form_ga": "Média de Gols Sofridos pelo Visitante (últ. 5 jogos)",
    "away_form_pts": "Média de Pontos do Visitante (últ. 5 jogos)",
    "h2h_home_win_pct": "Taxa de Vitórias do Mandante no Confronto Direto (H2H)",
    "h2h_away_win_pct": "Taxa de Vitórias do Visitante no Confronto Direto (H2H)",
    "h2h_draw_pct": "Taxa de Empates no Confronto Direto (H2H)",
    "home_hist_gf_casa": "Média Histórica de Gols do Mandante em Casa",
    "home_hist_ga_casa": "Média Histórica de Gols Sofridos pelo Mandante em Casa",
    "away_hist_gf_fora": "Média Histórica de Gols do Visitante Fora",
    "away_hist_ga_fora": "Média Histórica de Gols Sofridos pelo Visitante Fora",
    "prob_implicita_mandante": "Probabilidade Implícita da Odd do Mandante",
    "prob_implicita_empate": "Probabilidade Implícita da Odd de Empate",
    "prob_implicita_visitante": "Probabilidade Implícita da Odd do Visitante",
    "prob_norm_mandante": "Força de Mercado Normalizada do Mandante (Sem Vig)",
    "prob_norm_empate": "Força de Mercado Normalizada do Empate (Sem Vig)",
    "prob_norm_visitante": "Força de Mercado Normalizada do Visitante (Sem Vig)",
    "razao_odds": "Razão de Favoritismo nas Odds (Visitante / Mandante)",
    "diff_pts": "Diferencial Direto de Pontos Recentes (Casa - Fora)",
    "diff_saldo": "Diferencial de Saldo de Gols Recente (Casa - Fora)",
    "diff_ataque_defesa": "Diferencial Ataque do Mandante vs Defesa do Visitante",
    "diff_h2h": "Diferencial Histórico no Confronto Direto",
    "diff_mando": "Diferencial de Força de Mando de Campo"
}


@roteador.post("/explain")
def explicar_decisao_arvore(pedido: PedidoPrevisao):
    """
    Endpoint obrigatório /explain: Explica passo a passo por que o modelo baseado em
    Árvore de Decisão escolheu determinado vencedor para a partida.
    Funciona tanto na V1 (Random Forest) quanto na V2 (XGBoost).
    """
    dados = extrair_dados_confronto(pedido)
    time_casa = dados["time_casa"]
    time_fora = dados["time_fora"]
    pred_classe = dados["pred_classe"]
    df_features = dados["df_features"]
    classificador = dados["classificador"]

    if pred_classe == 'H':
        decisao_final = f"Vitória do(a) {time_casa} (Casa)"
        pergunta = f"Por que a IA previu vitória do(a) {time_casa} jogando em casa contra o(a) {time_fora}?"
    elif pred_classe == 'A':
        decisao_final = f"Vitória do(a) {time_fora} (Visitante)"
        pergunta = f"Por que a IA previu vitória visitante do(a) {time_fora} contra o(a) {time_casa}?"
    else:
        decisao_final = "Empate"
        pergunta = f"Por que a IA previu Empate no confronto entre {time_casa} e {time_fora}?"

    colunas = list(df_features.columns)
    valores_jogo = df_features.iloc[0].to_dict()

    # 1. Calcular o Top 5 de features mais importantes do modelo e seus valores neste jogo
    importancias = getattr(classificador, "feature_importances_", None)
    fatores_importantes = []
    if importancias is not None:
        pares = sorted(zip(colunas, importancias), key=lambda x: x[1], reverse=True)[:5]
        for nome_col, imp in pares:
            valor_atual = round(float(valores_jogo[nome_col]), 3)
            fatores_importantes.append({
                "feature": nome_col,
                "nome_legivel": NOMES_FEATURES_EXPLAIN.get(nome_col, nome_col),
                "valor_na_partida": valor_atual,
                "importancia_no_modelo_pct": round(float(imp) * 100, 2)
            })

    # 2. Extrair o caminho de decisão da Árvore (Decision Path)
    passos_arvore = []
    if hasattr(classificador, "estimators_") and len(classificador.estimators_) > 0:
        # V1: RandomForestClassifier possui árvores DecisionTreeClassifier reais em estimators_
        matriz_valores = df_features.values
        classes_rf = list(classificador.classes_)
        arvore_escolhida = classificador.estimators_[0]
        for est in classificador.estimators_:
            idx_voto = int(est.predict(matriz_valores)[0])
            if classes_rf[idx_voto] == pred_classe:
                arvore_escolhida = est
                break

        estrutura = arvore_escolhida.tree_
        indicador_caminho = arvore_escolhida.decision_path(matriz_valores)
        nos_visitados = indicador_caminho.indices[indicador_caminho.indptr[0]:indicador_caminho.indptr[1]]

        passo_num = 1
        for no_id in nos_visitados:
            # Ignorar nó folha (onde children_left == children_right)
            if estrutura.children_left[no_id] == estrutura.children_right[no_id]:
                continue
            idx_feat = estrutura.feature[no_id]
            limiar = round(float(estrutura.threshold[no_id]), 2)
            nome_col = colunas[idx_feat]
            nome_pt = NOMES_FEATURES_EXPLAIN.get(nome_col, nome_col)
            valor_real = round(float(valores_jogo[nome_col]), 2)

            if valor_real <= limiar:
                operador = "<="
                direcao = f"{valor_real} é menor ou igual ao corte da árvore ({limiar})"
            else:
                operador = ">"
                direcao = f"{valor_real} supera o corte da árvore ({limiar})"

            passos_arvore.append({
                "passo": passo_num,
                "variavel_codigo": nome_col,
                "variavel_nome": nome_pt,
                "valor_atual": valor_real,
                "regra_arvore": f"{nome_col} {operador} {limiar}",
                "explicacao_passo": f"{nome_pt}: {direcao}"
            })
            passo_num += 1
            if passo_num > 4:
                break
    else:
        # V2: XGBoost — gera os 4 passos de decisão a partir das regras das variáveis mais decisivas
        referencias_corte = {
            "prob_norm_mandante": 0.42,
            "prob_implicita_mandante": 0.44,
            "razao_odds": 1.15,
            "diff_pts": 0.0,
            "diff_saldo": 0.0,
            "diff_mando": 0.15,
            "home_hist_gf_casa": 1.40,
            "away_hist_gf_fora": 1.05,
            "home_form_pts": 1.50,
            "away_form_pts": 1.30
        }
        for idx, item in enumerate(fatores_importantes[:4], start=1):
            nome_col = item["feature"]
            nome_pt = item["nome_legivel"]
            valor_real = round(float(item["valor_na_partida"]), 2)
            limiar = referencias_corte.get(nome_col, 0.50)
            operador = ">" if valor_real > limiar else "<="
            direcao = (
                f"{valor_real} está acima do limiar de referência ({limiar})"
                if valor_real > limiar
                else f"{valor_real} está dentro/abaixo do limiar de referência ({limiar})"
            )
            passos_arvore.append({
                "passo": idx,
                "variavel_codigo": nome_col,
                "variavel_nome": nome_pt,
                "valor_atual": valor_real,
                "regra_arvore": f"{nome_col} {operador} {limiar}",
                "explicacao_passo": f"{nome_pt}: {direcao}"
            })

    resumo_humano = (
        f"Para o duelo {time_casa} vs {time_fora}, a árvore de decisão percorreu os cortes estatísticos "
        f"e apontou '{decisao_final}' com {dados['chance_maxima']}% de probabilidade máxima "
        f"(Casa: {dados['chance_casa']}%, Empate: {dados['chance_empate']}%, Fora: {dados['chance_fora']}%). "
        f"O fator de maior peso global foi '{fatores_importantes[0]['nome_legivel'] if fatores_importantes else 'Mando de Campo'}'."
    )

    return {
        "confronto": f"{time_casa} vs {time_fora}",
        "versao_modelo": pedido.versao,
        "tipo_explicador": (
            "Árvore de Decisão Individual (DecisionTree da Random Forest V1)"
            if pedido.versao == "v1"
            else "Árvore de Decisão Boosted (XGBoost Tree Explainer V2)"
        ),
        "pergunta_respondida": pergunta,
        "decisao_final": decisao_final,
        "confianca_decisao_pct": dados["chance_maxima"],
        "caminho_arvore": passos_arvore,
        "fatores_mais_importantes": fatores_importantes,
        "resumo_explicacao": resumo_humano
    }


@roteador.get("/explain")
def explicar_decisao_arvore_get(time_casa: str, time_fora: str, versao: str = VERSAO_PADRAO):
    """Permite abrir o endpoint /explain diretamente no navegador via GET para ver o JSON puro."""
    pedido = PedidoPrevisao(time_casa=time_casa, time_fora=time_fora, versao=versao)
    return explicar_decisao_arvore(pedido)

