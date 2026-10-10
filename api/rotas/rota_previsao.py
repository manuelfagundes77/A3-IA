from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from api.modelos import obter_classificador, obter_regressor, obter_estado

roteador = APIRouter()


class PedidoPrevisao(BaseModel):
    """Modelo do JSON que o frontend envia."""
    time_casa: str
    time_fora: str


def calcular_confianca(chance_maxima):
    """Retorna o nível de confiança baseado na maior probabilidade."""
    if chance_maxima > 60:
        return "Alta"
    elif chance_maxima >= 50:
        return "Media"
    else:
        return "Baixa"


@roteador.post("/api/previsao")
def fazer_previsao(pedido: PedidoPrevisao):
    """Recebe dois times e retorna a previsão completa com Raio-X."""
    classificador = obter_classificador()
    regressor = obter_regressor()
    estado = obter_estado()

    time_casa = pedido.time_casa
    time_fora = pedido.time_fora

    historico_times = estado['historico_times']
    historico_em_casa = estado['historico_em_casa']
    historico_fora = estado['historico_fora']
    historico_h2h = estado['historico_h2h']

    # --- Momento Atual (Últimos 5 jogos) ---
    home_games = historico_times.get(time_casa, [])[-5:]
    home_gf  = sum([g['gf'] for g in home_games]) / 5 if len(home_games) > 0 else 0
    home_ga  = sum([g['ga'] for g in home_games]) / 5 if len(home_games) > 0 else 0
    home_pts = sum([g['pts'] for g in home_games]) / 5 if len(home_games) > 0 else 0

    away_games = historico_times.get(time_fora, [])[-5:]
    away_gf  = sum([g['gf'] for g in away_games]) / 5 if len(away_games) > 0 else 0
    away_ga  = sum([g['ga'] for g in away_games]) / 5 if len(away_games) > 0 else 0
    away_pts = sum([g['pts'] for g in away_games]) / 5 if len(away_games) > 0 else 0

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
        home_hist_gf_casa, home_hist_ga_casa = 0.0, 0.0

    jogos_fora_hist = historico_fora.get(time_fora, [])
    if len(jogos_fora_hist) > 0:
        away_hist_gf_fora = sum([g['gf'] for g in jogos_fora_hist]) / len(jogos_fora_hist)
        away_hist_ga_fora = sum([g['ga'] for g in jogos_fora_hist]) / len(jogos_fora_hist)
    else:
        away_hist_gf_fora, away_hist_ga_fora = 0.0, 0.0

    # --- Montar features na mesma ordem do treino ---
    features = pd.DataFrame([{
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

    # --- Previsão ---
    pred_classe = classificador.predict(features)[0]
    probabilidades = classificador.predict_proba(features)[0]
    classes_modelo = classificador.classes_
    pred_gols = regressor.predict(features)[0]

    probs = dict(zip(classes_modelo, probabilidades))
    chance_casa   = round(probs.get('H', 0) * 100, 1)
    chance_empate = round(probs.get('D', 0) * 100, 1)
    chance_fora   = round(probs.get('A', 0) * 100, 1)

    chance_maxima = max(chance_casa, chance_empate, chance_fora)

    # Palpite em texto
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
                "gols_feitos_media": round(home_gf, 2),
                "gols_sofridos_media": round(home_ga, 2),
                "pontos_media": round(home_pts, 2)
            },
            "fora": {
                "gols_feitos_media": round(away_gf, 2),
                "gols_sofridos_media": round(away_ga, 2),
                "pontos_media": round(away_pts, 2)
            }
        },
        "retrospecto": {
            "vitorias_casa_pct": round(h2h_home_wins * 100, 1),
            "empates_pct": round(h2h_draws * 100, 1),
            "vitorias_fora_pct": round(h2h_away_wins * 100, 1)
        },
        "previsao": {
            "palpite": palpite,
            "chance_casa_pct": chance_casa,
            "chance_empate_pct": chance_empate,
            "chance_fora_pct": chance_fora,
            "total_gols_esperados": round(pred_gols, 2),
            "confianca": calcular_confianca(chance_maxima)
        }
    }

