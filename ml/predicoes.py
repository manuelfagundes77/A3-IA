import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import os
from config_versoes import VERSAO_ATIVA, pasta_modelos

def carregar_modelos():
    # Somente leitura: pode usar versões finalizadas sem risco
    pasta_mod = pasta_modelos(VERSAO_ATIVA)
    with open(os.path.join(pasta_mod, 'classificador_vencedor.pkl'), 'rb') as f:
        classificador = pickle.load(f)
        
    with open(os.path.join(pasta_mod, 'regressor_gols.pkl'), 'rb') as f:
        regressor = pickle.load(f)
        
    # Agora carregamos o histórico completo para extrair o momento mais recente
    with open(os.path.join(pasta_mod, 'historico_estado.pkl'), 'rb') as f:
        estado = pickle.load(f)
        
    return classificador, regressor, estado

def fazer_previsao(time_casa, time_fora, classificador, regressor, estado):
    historico_times = estado['historico_times']
    historico_h2h = estado['historico_h2h']
    historico_em_casa = estado['historico_em_casa']
    historico_fora = estado['historico_fora']
    
    # Buscar Momento Atual da Casa (Últimos 5 jogos)
    home_games = historico_times.get(time_casa, [])[-5:]
    home_gf = sum([g['gf'] for g in home_games]) / 5 if len(home_games) > 0 else 0
    home_ga = sum([g['ga'] for g in home_games]) / 5 if len(home_games) > 0 else 0
    home_pts = sum([g['pts'] for g in home_games]) / 5 if len(home_games) > 0 else 0
    
    # Buscar Momento Atual de Fora (Últimos 5 jogos)
    away_games = historico_times.get(time_fora, [])[-5:]
    away_gf = sum([g['gf'] for g in away_games]) / 5 if len(away_games) > 0 else 0
    away_ga = sum([g['ga'] for g in away_games]) / 5 if len(away_games) > 0 else 0
    away_pts = sum([g['pts'] for g in away_games]) / 5 if len(away_games) > 0 else 0
    
    # Buscar Confronto Direto (Últimos 5 jogos entre ELES)
    equipes = tuple(sorted([time_casa, time_fora]))
    h2h_games = historico_h2h.get(equipes, [])[-5:]
    
    if len(h2h_games) > 0:
        h2h_home_wins = sum([1 for g in h2h_games if g['winner'] == time_casa]) / len(h2h_games)
        h2h_away_wins = sum([1 for g in h2h_games if g['winner'] == time_fora]) / len(h2h_games)
        h2h_draws = sum([1 for g in h2h_games if g['winner'] == 'Draw']) / len(h2h_games)
    else:
        h2h_home_wins, h2h_away_wins, h2h_draws = 0.0, 0.0, 0.0

    # Buscar Média Histórica de Casa/Fora
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
        
    # Criar as features exatamente na ordem que o modelo aprendeu (13 features)
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
    
    # Prever
    pred_classe = classificador.predict(features)[0]
    probabilidades = classificador.predict_proba(features)[0]
    classes_modelo = classificador.classes_
    pred_gols = regressor.predict(features)[0]
    
    probs = dict(zip(classes_modelo, probabilidades))
    prob_casa = probs.get('H', 0) * 100
    prob_empate = probs.get('D', 0) * 100
    prob_fora = probs.get('A', 0) * 100
    
    if pred_classe == 'H':
        resultado_str = f"Vitória do(a) {time_casa} (Casa)"
    elif pred_classe == 'A':
        resultado_str = f"Vitória do(a) {time_fora} (Visitante)"
    else:
        resultado_str = "Empate"
        
    print("-" * 60)
    print(f">>> PREVISÃO: {time_casa} (Casa) vs {time_fora} (Visitante)")
    print("-" * 60)
    print(f"--- RAIO-X: Momento Atual (Últimos 5 jogos na liga)")
    print(f"  -> {time_casa}: Fez {home_gf:.1f} gols/jogo | Sofreu {home_ga:.1f} | Ganhou {home_pts:.1f} pts/jogo")
    print(f"  -> {time_fora}: Fez {away_gf:.1f} gols/jogo | Sofreu {away_ga:.1f} | Ganhou {away_pts:.1f} pts/jogo")
    print("")
    print(f"--- CONFRONTO DIRETO (Últimos 5 encontros)")
    print(f"  -> {h2h_home_wins*100:.0f}% de Vitórias do {time_casa}")
    print(f"  -> {h2h_draws*100:.0f}% de Empates")
    print(f"  -> {h2h_away_wins*100:.0f}% de Vitórias do {time_fora}")
    print("-" * 60)
    print(f"*** DECISÃO DA INTELIGÊNCIA ARTIFICIAL")
    print(f"  -> Chances: {time_casa} {prob_casa:.1f}% | Empate {prob_empate:.1f}% | {time_fora} {prob_fora:.1f}%")
    print(f"  -> Palpite Vencedor: {resultado_str}")
    print(f"  -> Expectativa de Gols: {pred_gols:.2f} gols no jogo")
    print("-" * 60)

def main():
    print("Carregando modelos e retrospecto atualizado...")
    try:
        classificador, regressor, estado = carregar_modelos()
    except FileNotFoundError:
        print("Modelos não encontrados. Execute 'python treino.py' primeiro para gerar os novos modelos.")
        return
        
    # Testes Manuais
    time_casa_1 = "Flamengo RJ"
    time_fora_1 = "Palmeiras"
    fazer_previsao(time_casa_1, time_fora_1, classificador, regressor, estado)
    
    time_casa_2 = "Sao Paulo"
    time_fora_2 = "Corinthians"
    fazer_previsao(time_casa_2, time_fora_2, classificador, regressor, estado)

if __name__ == '__main__':
    main()
