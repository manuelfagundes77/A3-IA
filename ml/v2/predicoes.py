import os
import pickle
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(PASTA_ATUAL, 'modelos')

from treino import extrair_features_linha


def carregar_modelos():
    """
    Carrega os modelos treinados pelo XGBoost e o estado histórico salvo.
    """
    caminho_classificador = os.path.join(PASTA_MODELOS, 'classificador_vencedor.pkl')
    caminho_regressor = os.path.join(PASTA_MODELOS, 'regressor_gols.pkl')
    caminho_estado = os.path.join(PASTA_MODELOS, 'historico_estado.pkl')

    if not (os.path.exists(caminho_classificador) and os.path.exists(caminho_regressor) and os.path.exists(caminho_estado)):
        raise FileNotFoundError(
            "Arquivos de modelos da V2 não encontrados. "
            "Por favor, execute 'python treino.py' dentro da pasta ml/v2 primeiro."
        )

    with open(caminho_classificador, 'rb') as f:
        classificador = pickle.load(f)

    with open(caminho_regressor, 'rb') as f:
        regressor = pickle.load(f)

    with open(caminho_estado, 'rb') as f:
        estado = pickle.load(f)

    return classificador, regressor, estado


def fazer_previsao(time_casa, time_fora, classificador, regressor, estado,
                   odd_casa=None, odd_empate=None, odd_fora=None):
    """
    Gera a previsão para uma partida utilizando as 25 features otimizadas da V2.
    """
    historico_times = estado['historico_times']
    historico_h2h = estado['historico_h2h']
    historico_em_casa = estado['historico_em_casa']
    historico_fora = estado['historico_fora']
    features_esperadas = estado.get('features', None)
    mapa_inverso = estado.get('mapa_inverso', {0: 'A', 1: 'D', 2: 'H'})

    odd_h = float(odd_casa) if (odd_casa and float(odd_casa) > 1.0) else 2.15
    odd_d = float(odd_empate) if (odd_empate and float(odd_empate) > 1.0) else 3.20
    odd_a = float(odd_fora) if (odd_fora and float(odd_fora) > 1.0) else 3.50

    dic_features = extrair_features_linha(
        time_casa, time_fora, historico_times, historico_em_casa, historico_fora,
        historico_h2h, odd_h, odd_d, odd_a
    )

    features = pd.DataFrame([dic_features])
    if features_esperadas:
        features = features[features_esperadas]

    # Predição com o XGBoost
    pred_classe_num = classificador.predict(features)[0]
    probabilidades = classificador.predict_proba(features)[0]
    pred_gols = float(regressor.predict(features)[0])

    pred_classe = mapa_inverso.get(pred_classe_num, 'H')

    # No XGBoost com classes [0, 1, 2] mapeadas para ['A', 'D', 'H']:
    prob_fora = round(float(probabilidades[0]) * 100, 1)
    prob_empate = round(float(probabilidades[1]) * 100, 1)
    prob_casa = round(float(probabilidades[2]) * 100, 1)

    prob_maxima = max(prob_casa, prob_empate, prob_fora)
    if prob_maxima > 60:
        confianca = "Alta"
    elif prob_maxima >= 50:
        confianca = "Média"
    else:
        confianca = "Baixa"

    if pred_classe == 'H':
        resultado_str = f"Vitória do(a) {time_casa} (Mandante)"
    elif pred_classe == 'A':
        resultado_str = f"Vitória do(a) {time_fora} (Visitante)"
    else:
        resultado_str = "Empate"

    # Estimativa de placar provável
    gols_esperados = max(0.0, pred_gols)
    gols_casa_est = round(max(0.0, (gols_esperados * (prob_casa / 100)) + 0.3))
    gols_fora_est = round(max(0.0, (gols_esperados * (prob_fora / 100))))
    placar_provavel = f"{int(gols_casa_est)} x {int(gols_fora_est)}"

    return {
        'time_casa': time_casa,
        'time_fora': time_fora,
        'vencedor_previsto': resultado_str,
        'codigo_vencedor': pred_classe,
        'probabilidades': {
            'casa': prob_casa,
            'empate': prob_empate,
            'fora': prob_fora,
        },
        'expectativa_gols': round(gols_esperados, 2),
        'placar_provavel': placar_provavel,
        'confianca': confianca,
        'confianca_pct': prob_maxima,
        'odds_utilizadas': {
            'casa': odd_h,
            'empate': odd_d,
            'fora': odd_a,
        },
        'raio_x': {
            'casa': {
                'gf': dic_features['home_form_gf'],
                'ga': dic_features['home_form_ga'],
                'pts': dic_features['home_form_pts'],
                'media_casa_gf': dic_features['home_hist_gf_casa']
            },
            'fora': {
                'gf': dic_features['away_form_gf'],
                'ga': dic_features['away_form_ga'],
                'pts': dic_features['away_form_pts'],
                'media_fora_gf': dic_features['away_hist_gf_fora']
            },
            'diferenciais': {
                'diff_pts': dic_features['diff_pts'],
                'diff_saldo': dic_features['diff_saldo'],
                'razao_odds': dic_features['razao_odds']
            }
        }
    }


def main():
    print("=" * 60)
    print("  ML V2 (XGBoost Otimizado) - Simulador com 25 Features")
    print("=" * 60)

    try:
        classificador, regressor, estado = carregar_modelos()
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        return

    times_disponiveis = sorted(list(estado['historico_times'].keys()))
    print(f"Total de {len(times_disponiveis)} times disponíveis no histórico.")
    print("Exemplos: Flamengo RJ, Palmeiras, Sao Paulo, Corinthians, Gremio, etc.")
    print("-" * 60)

    time_casa = input("Time Mandante (Casa): ").strip()
    time_fora = input("Time Visitante (Fora): ").strip()

    if time_casa not in estado['historico_times']:
        print(f"Aviso: '{time_casa}' não encontrado no histórico. Usando baseline padrão.")
    if time_fora not in estado['historico_times']:
        print(f"Aviso: '{time_fora}' não encontrado no histórico. Usando baseline padrão.")

    print("\n(Opcional) Digite as odds das casas de apostas (pressione Enter para usar média):")
    odd_h_str = input(f"Odd {time_casa} [Padrão 2.15]: ").strip()
    odd_d_str = input("Odd Empate [Padrão 3.20]: ").strip()
    odd_a_str = input(f"Odd {time_fora} [Padrão 3.50]: ").strip()

    odd_h = float(odd_h_str) if odd_h_str else None
    odd_d = float(odd_d_str) if odd_d_str else None
    odd_a = float(odd_a_str) if odd_a_str else None

    resultado = fazer_previsao(time_casa, time_fora, classificador, regressor, estado, odd_h, odd_d, odd_a)

    print("\n" + "=" * 60)
    print(f"  PREVISÃO: {time_casa} vs {time_fora}")
    print("=" * 60)
    print(f"  Palpite do Vencedor:  {resultado['vencedor_previsto']}")
    print(f"  Grau de Confiança:     {resultado['confianca']} ({resultado['confianca_pct']}%)")
    print(f"  Placar Mais Provável:  {resultado['placar_provavel']}")
    print(f"  Expectativa de Gols:   {resultado['expectativa_gols']} gols")
    print("-" * 60)
    print("  Probabilidades Calculadas pelo XGBoost:")
    print(f"  • Vitória do Mandante ({time_casa}):  {resultado['probabilidades']['casa']}%")
    print(f"  • Empate:                               {resultado['probabilidades']['empate']}%")
    print(f"  • Vitória do Visitante ({time_fora}): {resultado['probabilidades']['fora']}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
