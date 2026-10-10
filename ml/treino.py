import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import pickle
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calcular_features_e_historico(df):
    times = pd.concat([df['Home'], df['Away']]).unique()

    # Histórico geral (últimos 5 jogos - qualquer mando)
    historico_times = {time: [] for time in times}

    # Histórico separado por mando de campo (para média histórica de casa/fora)
    historico_times_em_casa = {time: [] for time in times}
    historico_times_fora    = {time: [] for time in times}

    # Histórico de confronto direto
    historico_h2h = {}

    features_list = []

    for index, row in df.iterrows():
        home = row['Home']
        away = row['Away']
        hg = row['HG']
        ag = row['AG']

        equipes = tuple(sorted([home, away]))
        if equipes not in historico_h2h:
            historico_h2h[equipes] = []

        # --- FEATURE 1: Momento Atual (Últimos 5 jogos de cada time) ---
        home_games = historico_times[home][-5:]
        home_gf  = sum([g['gf'] for g in home_games]) / 5 if len(home_games) > 0 else 0
        home_ga  = sum([g['ga'] for g in home_games]) / 5 if len(home_games) > 0 else 0
        home_pts = sum([g['pts'] for g in home_games]) / 5 if len(home_games) > 0 else 0

        away_games = historico_times[away][-5:]
        away_gf  = sum([g['gf'] for g in away_games]) / 5 if len(away_games) > 0 else 0
        away_ga  = sum([g['ga'] for g in away_games]) / 5 if len(away_games) > 0 else 0
        away_pts = sum([g['pts'] for g in away_games]) / 5 if len(away_games) > 0 else 0

        # --- FEATURE 2: Confronto Direto (Últimos 5 jogos entre ELES) ---
        h2h_games = historico_h2h[equipes][-5:]
        if len(h2h_games) > 0:
            h2h_home_wins = sum([1 for g in h2h_games if g['winner'] == home]) / len(h2h_games)
            h2h_away_wins = sum([1 for g in h2h_games if g['winner'] == away]) / len(h2h_games)
            h2h_draws     = sum([1 for g in h2h_games if g['winner'] == 'Draw']) / len(h2h_games)
        else:
            h2h_home_wins, h2h_away_wins, h2h_draws = 0.0, 0.0, 0.0

        # --- FEATURE 3: Média Histórica de Casa/Fora (NOVA) ---
        # Mandante: como ele se sai jogando EM CASA historicamente
        jogos_em_casa = historico_times_em_casa[home]
        if len(jogos_em_casa) > 0:
            home_hist_gf_casa = sum([g['gf'] for g in jogos_em_casa]) / len(jogos_em_casa)
            home_hist_ga_casa = sum([g['ga'] for g in jogos_em_casa]) / len(jogos_em_casa)
        else:
            home_hist_gf_casa = 0.0
            home_hist_ga_casa = 0.0

        # Visitante: como ele se sai jogando FORA historicamente
        jogos_fora = historico_times_fora[away]
        if len(jogos_fora) > 0:
            away_hist_gf_fora = sum([g['gf'] for g in jogos_fora]) / len(jogos_fora)
            away_hist_ga_fora = sum([g['ga'] for g in jogos_fora]) / len(jogos_fora)
        else:
            away_hist_gf_fora = 0.0
            away_hist_ga_fora = 0.0

        features_list.append({
            # Momento atual (últimos 5 jogos)
            'home_form_gf': home_gf,
            'home_form_ga': home_ga,
            'home_form_pts': home_pts,
            'away_form_gf': away_gf,
            'away_form_ga': away_ga,
            'away_form_pts': away_pts,
            # Confronto direto
            'h2h_home_win_pct': h2h_home_wins,
            'h2h_away_win_pct': h2h_away_wins,
            'h2h_draw_pct': h2h_draws,
            # Médias históricas por mando (NOVAS)
            'home_hist_gf_casa': home_hist_gf_casa,
            'home_hist_ga_casa': home_hist_ga_casa,
            'away_hist_gf_fora': away_hist_gf_fora,
            'away_hist_ga_fora': away_hist_ga_fora,
        })

        # --- ATUALIZAR HISTÓRICO APÓS O JOGO ---
        if hg > ag:
            h_pts, a_pts, winner = 3, 0, home
        elif hg < ag:
            h_pts, a_pts, winner = 0, 3, away
        else:
            h_pts, a_pts, winner = 1, 1, 'Draw'

        historico_times[home].append({'gf': hg, 'ga': ag, 'pts': h_pts})
        historico_times[away].append({'gf': ag, 'ga': hg, 'pts': a_pts})

        # Atualizar histórico separado por mando de campo
        historico_times_em_casa[home].append({'gf': hg, 'ga': ag})
        historico_times_fora[away].append({'gf': ag, 'ga': hg})

        historico_h2h[equipes].append({'winner': winner})

    df_feat = pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1)
    return df_feat, historico_times, historico_times_em_casa, historico_times_fora, historico_h2h


def main():
    print("=" * 55)
    print("  Treinamento V1 - Com Medias Historicas de Casa/Fora")
    print("=" * 55)

    dados = pd.read_csv('dados/BRA.csv')
    dados.dropna(subset=['Season', 'Date', 'Home', 'Away', 'HG', 'AG', 'Res'], inplace=True)
    dados['Date'] = pd.to_datetime(dados['Date'], format='%d/%m/%Y')
    dados = dados.sort_values('Date')

    print("Calculando features jogo a jogo... (Pode levar alguns segundos)")
    dados_com_features, historico_times, historico_em_casa, historico_fora, historico_h2h = calcular_features_e_historico(dados)

    dados_treino = dados_com_features[dados_com_features['Season'] < 2024].copy()
    dados_teste  = dados_com_features[dados_com_features['Season'] >= 2024].copy()

    features = [
        # Momento atual
        'home_form_gf', 'home_form_ga', 'home_form_pts',
        'away_form_gf', 'away_form_ga', 'away_form_pts',
        # Confronto direto
        'h2h_home_win_pct', 'h2h_away_win_pct', 'h2h_draw_pct',
        # Médias históricas por mando (NOVAS)
        'home_hist_gf_casa', 'home_hist_ga_casa',
        'away_hist_gf_fora', 'away_hist_ga_fora',
    ]

    dados_treino['total_gols'] = dados_treino['HG'] + dados_treino['AG']
    dados_teste['total_gols']  = dados_teste['HG']  + dados_teste['AG']

    X_treino       = dados_treino[features]
    y_class_treino = dados_treino['Res']
    y_reg_treino   = dados_treino['total_gols']

    print("Treinando Random Forest com 13 features...")
    classificador_vencedor = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    classificador_vencedor.fit(X_treino, y_class_treino)

    regressor_gols = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
    regressor_gols.fit(X_treino, y_reg_treino)

    # Salvar modelos e histórico completo
    import os
    os.makedirs('modelos_v1', exist_ok=True)
    with open('modelos_v1/classificador_vencedor.pkl', 'wb') as f:
        pickle.dump(classificador_vencedor, f)
    with open('modelos_v1/regressor_gols.pkl', 'wb') as f:
        pickle.dump(regressor_gols, f)
    with open('modelos_v1/historico_estado.pkl', 'wb') as f:
        pickle.dump({
            'historico_times': historico_times,
            'historico_em_casa': historico_em_casa,
            'historico_fora': historico_fora,
            'historico_h2h': historico_h2h
        }, f)

    print("\nModelos e historico salvos com sucesso!")

    # Métricas no Treino
    pred_class_treino = classificador_vencedor.predict(X_treino)
    pred_reg_treino   = regressor_gols.predict(X_treino)
    acc_treino  = accuracy_score(y_class_treino, pred_class_treino) * 100
    erro_treino = mean_absolute_error(y_reg_treino, pred_reg_treino)

    # Métricas no Teste Real (2024+)
    X_teste       = dados_teste[features]
    y_class_teste = dados_teste['Res']
    y_reg_teste   = dados_teste['total_gols']

    pred_class_teste = classificador_vencedor.predict(X_teste)
    pred_reg_teste   = regressor_gols.predict(X_teste)
    acc_teste  = accuracy_score(y_class_teste, pred_class_teste) * 100
    erro_teste = mean_absolute_error(y_reg_teste, pred_reg_teste)

    print("\n" + "=" * 55)
    print("  RESULTADOS (Compare com a versao anterior!)")
    print("=" * 55)
    print(f"  Features usadas: 9 anteriores + 4 novas = 13 total")
    print(f"\n  --- Treino (2012-2023) ---")
    print(f"  Acuracia (Quem Vence): {acc_treino:.2f}%")
    print(f"  Erro Medio (Total de Gols): {erro_treino:.2f} gols")
    print(f"\n  --- Teste Real (2024 em diante) ---")
    print(f"  Acuracia (Quem Vence): {acc_teste:.2f}%  (era 48.32%)")
    print(f"  Erro Medio (Total de Gols): {erro_teste:.2f} gols  (era 1.20)")
    print(f"\n  Diferenca na Acuracia: {acc_teste - 48.32:+.2f}%")
    print(f"  Diferenca no Erro de Gols: {erro_teste - 1.20:+.2f} gols")
    print("=" * 55)

    # Salvar resultados para graficos
    resultados = pd.DataFrame({
        'classe_real': y_class_teste,
        'classe_prevista': pred_class_teste,
        'gols_reais': y_reg_teste,
        'gols_previstos': pred_reg_teste
    })
    resultados.to_csv('resultados_2024.csv', index=False)
    print("\nArquivo resultados_2024.csv atualizado.")

    # Gerar gráficos automáticos da fase de treinamento na pasta graficos_treino_v1
    gerar_graficos_treino(
        classificador=classificador_vencedor,
        regressor=regressor_gols,
        features=features,
        acc_treino=acc_treino,
        acc_teste=acc_teste,
        erro_treino=erro_treino,
        erro_teste=erro_teste,
        y_reg_treino=y_reg_treino,
        pred_reg_treino=pred_reg_treino
    )


MAPA_NOMES_FEATURES = {
    'home_form_gf': 'Gols Feitos Casa (5j)',
    'home_form_ga': 'Gols Sofridos Casa (5j)',
    'home_form_pts': 'Pontos Casa (5j)',
    'away_form_gf': 'Gols Feitos Fora (5j)',
    'away_form_ga': 'Gols Sofridos Fora (5j)',
    'away_form_pts': 'Pontos Fora (5j)',
    'h2h_home_win_pct': 'Vitorias Casa no H2H',
    'h2h_away_win_pct': 'Vitorias Fora no H2H',
    'h2h_draw_pct': 'Empates no H2H',
    'home_hist_gf_casa': 'Historico Gols Pro (Casa)',
    'home_hist_ga_casa': 'Historico Gols Contra (Casa)',
    'away_hist_gf_fora': 'Historico Gols Pro (Fora)',
    'away_hist_ga_fora': 'Historico Gols Contra (Fora)',
}


def gerar_graficos_treino(classificador, regressor, features, acc_treino, acc_teste, erro_treino, erro_teste, y_reg_treino, pred_reg_treino):
    pasta = 'graficos_treino_v1'
    os.makedirs(pasta, exist_ok=True)
    print(f"\n" + "=" * 55)
    print(f"  Gerando Graficos de Treinamento em '{pasta}/'...")
    print("=" * 55)

    estilo_escuro = {
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'axes.titlecolor': 'white',
        'axes.edgecolor': '#0f3460'
    }
    plt.rcParams.update(estilo_escuro)

    # 1. Importancia das 13 Features
    importancias = classificador.feature_importances_
    indices = np.argsort(importancias)
    nomes_bonitos = [MAPA_NOMES_FEATURES.get(features[i], features[i]) for i in indices]
    valores_ordenados = importancias[indices] * 100

    plt.figure(figsize=(10, 6))
    barras = plt.barh(range(len(indices)), valores_ordenados, color='#e94560', edgecolor='white', linewidth=0.5)
    plt.yticks(range(len(indices)), nomes_bonitos, fontsize=9)
    plt.xlabel('Importancia Relativa (%)', fontsize=11, color='white')
    plt.title('Importancia das 13 Features no Treinamento (Random Forest)', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    for barra, valor in zip(barras, valores_ordenados):
        plt.text(barra.get_width() + 0.2, barra.get_y() + barra.get_height()/2,
                 f'{valor:.1f}%', va='center', fontsize=9, color='white', fontweight='bold')

    plt.xlim(0, max(valores_ordenados) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, '1_importancia_features.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("  -> Salvo: 1_importancia_features.png")

    # 2. Treino vs Teste Real (Diagnostico de Overfitting)
    plt.figure(figsize=(8, 6))
    categorias = ['Chute Aleatorio\n(Baseline)', 'Teste Real (2024+)\n(Generalizacao)', 'Treino (2012-2023)\n(Aprendizado)']
    valores_acc = [33.33, acc_teste, acc_treino]
    cores = ['#0f3460', '#4ade80', '#e94560']

    barras_acc = plt.bar(categorias, valores_acc, color=cores, width=0.45, edgecolor='white', linewidth=0.8)
    plt.ylabel('Acuracia (%)', fontsize=11, color='white')
    plt.title('Avaliacao de Overfitting: Treino vs Teste Real', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 75)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for barra, valor in zip(barras_acc, valores_acc):
        plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1.2,
                 f'{valor:.2f}%', ha='center', fontsize=11, fontweight='bold', color='white')

    diff_overfit = acc_treino - acc_teste
    plt.text(0.5, 0.88, f"Gap de Overfitting: {diff_overfit:.2f}% (< 10% = Saudavel)",
             ha='center', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460', edgecolor='#4ade80', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(pasta, '2_treino_vs_validacao.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("  -> Salvo: 2_treino_vs_validacao.png")

    # 3. Diagnostico do Aprendizado de Gols
    plt.figure(figsize=(9, 5))
    erros_gols_treino = pred_reg_treino - y_reg_treino

    plt.hist(erros_gols_treino, bins=25, color='#38bdf8', edgecolor='white', linewidth=0.6, alpha=0.85)
    plt.axvline(0, color='#e94560', linestyle='--', linewidth=1.5, label='Erro Zero (Previsao Perfeita)')
    plt.title(f'Diagnostico de Gols no Treino: Erro Medio = {erro_treino:.2f} gols', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Erro de Previsao (Gols Previstos - Gols Reais)', fontsize=11, color='white')
    plt.ylabel('Frequencia de Partidas', fontsize=11, color='white')
    plt.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(pasta, '3_diagnostico_gols_treino.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("  -> Salvo: 3_diagnostico_gols_treino.png")
    print("=" * 55)


if __name__ == '__main__':
    main()
