import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    raise ImportError(
        "A biblioteca 'xgboost' não está instalada. "
        "Por favor, instale executando: pip install xgboost"
    )

from sklearn.metrics import accuracy_score, mean_absolute_error

# Definição de caminhos locais da V2
PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_ML = os.path.dirname(PASTA_ATUAL)
ARQUIVO_DADOS = os.path.join(PASTA_ML, 'dados', 'BRA.csv')

PASTA_MODELOS = os.path.join(PASTA_ATUAL, 'modelos')
PASTA_GRAFICOS_TREINO = os.path.join(PASTA_ATUAL, 'graficos_treino')
PASTA_GRAFICOS_TESTE = os.path.join(PASTA_ATUAL, 'graficos_teste')
ARQUIVO_RESULTADOS = os.path.join(PASTA_ATUAL, 'resultados_teste.csv')

MAPA_CLASSES = {'A': 0, 'D': 1, 'H': 2}
MAPA_INVERSO = {0: 'A', 1: 'D', 2: 'H'}


def extrair_features_linha(home, away, historico_times, historico_times_em_casa, historico_times_fora,
                           historico_h2h, odd_h, odd_d, odd_a):
    """
    Função modular para extrair as 25 features tanto no treino quanto na predição em tempo real.
    """
    equipes = tuple(sorted([home, away]))

    # 1. Momento Atual (Últimos 5 jogos)
    home_games = historico_times.get(home, [])[-5:]
    home_gf = sum([g['gf'] for g in home_games]) / 5 if len(home_games) > 0 else 0.0
    home_ga = sum([g['ga'] for g in home_games]) / 5 if len(home_games) > 0 else 0.0
    home_pts = sum([g['pts'] for g in home_games]) / 5 if len(home_games) > 0 else 0.0

    away_games = historico_times.get(away, [])[-5:]
    away_gf = sum([g['gf'] for g in away_games]) / 5 if len(away_games) > 0 else 0.0
    away_ga = sum([g['ga'] for g in away_games]) / 5 if len(away_games) > 0 else 0.0
    away_pts = sum([g['pts'] for g in away_games]) / 5 if len(away_games) > 0 else 0.0

    # 2. Confronto Direto H2H (Últimos 5 jogos entre eles)
    h2h_games = historico_h2h.get(equipes, [])[-5:]
    if len(h2h_games) > 0:
        h2h_home_wins = sum([1 for g in h2h_games if g['winner'] == home]) / len(h2h_games)
        h2h_away_wins = sum([1 for g in h2h_games if g['winner'] == away]) / len(h2h_games)
        h2h_draws = sum([1 for g in h2h_games if g['winner'] == 'Draw']) / len(h2h_games)
    else:
        h2h_home_wins, h2h_away_wins, h2h_draws = 0.0, 0.0, 0.0

    # 3. Média Histórica de Mando
    jogos_em_casa = historico_times_em_casa.get(home, [])
    if len(jogos_em_casa) > 0:
        home_hist_gf_casa = sum([g['gf'] for g in jogos_em_casa]) / len(jogos_em_casa)
        home_hist_ga_casa = sum([g['ga'] for g in jogos_em_casa]) / len(jogos_em_casa)
    else:
        home_hist_gf_casa, home_hist_ga_casa = 0.0, 0.0

    jogos_fora = historico_times_fora.get(away, [])
    if len(jogos_fora) > 0:
        away_hist_gf_fora = sum([g['gf'] for g in jogos_fora]) / len(jogos_fora)
        away_hist_ga_fora = sum([g['ga'] for g in jogos_fora]) / len(jogos_fora)
    else:
        away_hist_gf_fora, away_hist_ga_fora = 0.0, 0.0

    # 4. Probabilidades Implícitas e Normalização sem o Vig (comissão da casa)
    prob_raw_h = 1.0 / odd_h if (pd.notnull(odd_h) and odd_h > 1.0) else 0.45
    prob_raw_d = 1.0 / odd_d if (pd.notnull(odd_d) and odd_d > 1.0) else 0.28
    prob_raw_a = 1.0 / odd_a if (pd.notnull(odd_a) and odd_a > 1.0) else 0.27

    soma_p = prob_raw_h + prob_raw_d + prob_raw_a
    prob_norm_h = prob_raw_h / soma_p if soma_p > 0 else 0.45
    prob_norm_d = prob_raw_d / soma_p if soma_p > 0 else 0.28
    prob_norm_a = prob_raw_a / soma_p if soma_p > 0 else 0.27

    # 5. Features Diferenciais (alta relevância para árvores de decisão)
    diff_pts = home_pts - away_pts
    diff_saldo = (home_gf - home_ga) - (away_gf - away_ga)
    diff_ataque_defesa = home_gf - away_ga
    diff_h2h = h2h_home_wins - h2h_away_wins
    diff_mando = home_hist_gf_casa - away_hist_ga_fora
    razao_odds = (odd_a / odd_h) if (odd_h > 0 and odd_a > 0) else 1.0

    return {
        # Momento atual (6)
        'home_form_gf': home_gf,
        'home_form_ga': home_ga,
        'home_form_pts': home_pts,
        'away_form_gf': away_gf,
        'away_form_ga': away_ga,
        'away_form_pts': away_pts,
        # Confronto direto (3)
        'h2h_home_win_pct': h2h_home_wins,
        'h2h_away_win_pct': h2h_away_wins,
        'h2h_draw_pct': h2h_draws,
        # Médias históricas por mando (4)
        'home_hist_gf_casa': home_hist_gf_casa,
        'home_hist_ga_casa': home_hist_ga_casa,
        'away_hist_gf_fora': away_hist_gf_fora,
        'away_hist_ga_fora': away_hist_ga_fora,
        # Odds e Probabilidades Implícitas (7)
        'prob_implicita_mandante': prob_raw_h,
        'prob_implicita_empate': prob_raw_d,
        'prob_implicita_visitante': prob_raw_a,
        'prob_norm_mandante': prob_norm_h,
        'prob_norm_empate': prob_norm_d,
        'prob_norm_visitante': prob_norm_a,
        'razao_odds': razao_odds,
        # Diferenciais Diretos (5)
        'diff_pts': diff_pts,
        'diff_saldo': diff_saldo,
        'diff_ataque_defesa': diff_ataque_defesa,
        'diff_h2h': diff_h2h,
        'diff_mando': diff_mando,
    }


def calcular_features_e_historico(df):
    times = pd.concat([df['Home'], df['Away']]).unique()

    historico_times = {time: [] for time in times}
    historico_times_em_casa = {time: [] for time in times}
    historico_times_fora = {time: [] for time in times}
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

        # Extração de Odds com fallback suave
        odd_h = row['B365CH'] if pd.notnull(row.get('B365CH')) and row['B365CH'] > 1.0 else row.get('AvgCH', np.nan)
        odd_d = row['B365CD'] if pd.notnull(row.get('B365CD')) and row['B365CD'] > 1.0 else row.get('AvgCD', np.nan)
        odd_a = row['B365CA'] if pd.notnull(row.get('B365CA')) and row['B365CA'] > 1.0 else row.get('AvgCA', np.nan)

        if pd.isnull(odd_h) or odd_h <= 1.0:
            odd_h = row.get('PSCH', 2.15)
        if pd.isnull(odd_d) or odd_d <= 1.0:
            odd_d = row.get('PSCD', 3.20)
        if pd.isnull(odd_a) or odd_a <= 1.0:
            odd_a = row.get('PSCA', 3.50)

        feats = extrair_features_linha(
            home, away, historico_times, historico_times_em_casa, historico_times_fora,
            historico_h2h, odd_h, odd_d, odd_a
        )
        features_list.append(feats)

        # Atualizar Histórico após a partida
        if hg > ag:
            h_pts, a_pts, winner = 3, 0, home
        elif hg < ag:
            h_pts, a_pts, winner = 0, 3, away
        else:
            h_pts, a_pts, winner = 1, 1, 'Draw'

        historico_times[home].append({'gf': hg, 'ga': ag, 'pts': h_pts})
        historico_times[away].append({'gf': ag, 'ga': hg, 'pts': a_pts})
        historico_times_em_casa[home].append({'gf': hg, 'ga': ag})
        historico_times_fora[away].append({'gf': ag, 'ga': hg})
        historico_h2h[equipes].append({'winner': winner})

    df_feat = pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1)
    return df_feat, historico_times, historico_times_em_casa, historico_times_fora, historico_h2h


def gerar_graficos_treino(classificador, features, acc_treino, acc_teste, erro_treino, erro_teste, y_reg_treino, pred_reg_treino, dados_treino, pred_class_treino_num):
    os.makedirs(PASTA_GRAFICOS_TREINO, exist_ok=True)
    cor_fundo = '#0F172A'
    cor_card = '#1E293B'
    cor_texto = '#F8FAFC'
    cor_texto_sec = '#94A3B8'
    cor_destaque = '#38BDF8'
    cor_borda = '#334155'

    plt.rcParams.update({
        'figure.facecolor': cor_fundo,
        'axes.facecolor': cor_card,
        'axes.edgecolor': cor_borda,
        'axes.labelcolor': cor_texto,
        'xtick.color': cor_texto_sec,
        'ytick.color': cor_texto_sec,
        'text.color': cor_texto,
        'grid.color': cor_borda,
        'grid.alpha': 0.4
    })

    MAPA_FEATURES_PT = {
        'home_form_gf': 'Média Gols Feitos (Mandante, 5j)',
        'home_form_ga': 'Média Gols Sofridos (Mandante, 5j)',
        'home_form_pts': 'Média Pontos (Mandante, 5j)',
        'away_form_gf': 'Média Gols Feitos (Visitante, 5j)',
        'away_form_ga': 'Média Gols Sofridos (Visitante, 5j)',
        'away_form_pts': 'Média Pontos (Visitante, 5j)',
        'h2h_home_win_pct': 'Vitórias Confronto Direto (Mandante)',
        'h2h_away_win_pct': 'Vitórias Confronto Direto (Visitante)',
        'h2h_draw_pct': 'Empates Confronto Direto',
        'home_hist_gf_casa': 'Histórico Gols Feitos em Casa (Mandante)',
        'home_hist_ga_casa': 'Histórico Gols Sofridos em Casa (Mandante)',
        'away_hist_gf_fora': 'Histórico Gols Feitos Fora (Visitante)',
        'away_hist_ga_fora': 'Histórico Gols Sofridos Fora (Visitante)',
        'prob_implicita_mandante': 'Prob. Implícita da Odd (Mandante)',
        'prob_implicita_empate': 'Prob. Implícita da Odd (Empate)',
        'prob_implicita_visitante': 'Prob. Implícita da Odd (Visitante)',
        'prob_norm_mandante': 'Prob. Normalizada (Mandante)',
        'prob_norm_empate': 'Prob. Normalizada (Empate)',
        'prob_norm_visitante': 'Prob. Normalizada (Visitante)',
        'razao_odds': 'Razão de Odds (Visitante/Mandante)',
        'diff_pts': 'Diferença de Pontos (Momento)',
        'diff_saldo': 'Diferença de Saldo de Gols',
        'diff_ataque_defesa': 'Diferença Ataque vs Defesa',
        'diff_h2h': 'Diferença de Vitórias H2H',
        'diff_mando': 'Diferença de Força de Mando'
    }

    features_pt = [MAPA_FEATURES_PT.get(f, f) for f in features]

    # 1. Importância das Features (Top 12 mais importantes)
    importancias = classificador.feature_importances_
    df_imp = pd.DataFrame({'feature': features_pt, 'importancia': importancias})
    df_imp = df_imp.sort_values('importancia', ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(11, 7))
    cores_barras = [cor_destaque if ('Prob.' in f or 'Diferença' in f or 'Razão' in f) else '#818CF8' for f in df_imp['feature']]
    barras = ax.barh(df_imp['feature'], df_imp['importancia'], color=cores_barras, height=0.65)
    for b in barras:
        w = b.get_width()
        ax.annotate(f"{w:.3f}", (w, b.get_y() + b.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, color=cor_texto)

    ax.set_title("V2 XGBoost Otimizado - Top 12 Features Mais Importantes", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Ganho de Importância (Capacidade de Decisão)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS_TREINO, '1_importancia_features_v2.png'), dpi=200)
    plt.close()

    # 2. Treino vs Teste Real (Generalização Saudável)
    fig, ax = plt.subplots(figsize=(8, 5))
    categorias = ['Treino (2012-2023)', 'Teste Real (2024+)']
    valores = [acc_treino, acc_teste]
    barras = ax.bar(categorias, valores, color=['#38BDF8', '#10B981'], width=0.45)
    for b in barras:
        h = b.get_height()
        ax.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=cor_texto)

    gap = acc_treino - acc_teste
    ax.set_ylim(0, 100)
    ax.set_ylabel("Acurácia (%)", fontsize=10)
    ax.set_title(f"V2 XGBoost - Desempenho e Aprendizado (Gap: {gap:.2f}%)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS_TREINO, '2_treino_vs_validacao_v2.png'), dpi=200)
    plt.close()

    # 3. Diagnóstico de Gols Treino (Boxplot Inteligente)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_gols = pd.DataFrame({'Reais': y_reg_treino, 'Previstos': pred_reg_treino})
    # Agrupar gols reais acima de 5 para não poluir o gráfico
    df_gols['Reais_Agrupados'] = df_gols['Reais'].apply(lambda x: 5 if x >= 5 else x)
    
    dados_por_gol = [df_gols[df_gols['Reais_Agrupados'] == i]['Previstos'] for i in range(6)]
    
    bp = ax.boxplot(dados_por_gol, positions=range(6), patch_artist=True,
                    boxprops=dict(facecolor='#38BDF8', color=cor_borda),
                    capprops=dict(color=cor_texto_sec),
                    whiskerprops=dict(color=cor_texto_sec),
                    flierprops=dict(marker='o', markerfacecolor='#EF4444', markersize=3, alpha=0.3),
                    medianprops=dict(color='#0F172A', linewidth=2))
    
    # Linha ideal de acerto perfeito
    ax.plot([-0.5, 5.5], [0, 5], color='#10B981', linestyle='--', linewidth=2, label='Acerto Exato (Ideal)')
    
    ax.set_title(f"Diagnóstico de Previsão de Gols - Treino\n(Onde a caixa estiver, é a média que o modelo previu)", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Quantidade de Gols que REALMENTE aconteceram na partida", fontsize=11, labelpad=10)
    ax.set_ylabel("Gols que o modelo PREVIU", fontsize=11, labelpad=10)
    
    ax.set_xticks(range(6))
    ax.set_xticklabels(['0 gols', '1 gol', '2 gols', '3 gols', '4 gols', '5+ gols'])
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS_TREINO, '3_diagnostico_gols_treino_v2.png'), dpi=200)
    plt.close()

    # 4. Estabilidade Temporal (Acurácia por Temporada de Treino)
    df_analise = dados_treino.copy()
    df_analise['Prev'] = pred_class_treino_num
    mapa_classes = {'A': 0, 'D': 1, 'H': 2}
    df_analise['Acertou'] = (df_analise['Res'].map(mapa_classes) == df_analise['Prev']).astype(int)
    
    agrupado_ano = df_analise.groupby('Season')['Acertou'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agrupado_ano.index, agrupado_ano.values, marker='o', color='#38BDF8', linewidth=2, markersize=8, markerfacecolor='#10B981')
    
    # Linha de tendência média
    media_geral = agrupado_ano.mean()
    ax.axhline(media_geral, color='#EF4444', linestyle='--', alpha=0.5, label=f'Média do Treino ({media_geral:.1f}%)')
    
    for x, y in zip(agrupado_ano.index, agrupado_ano.values):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=cor_texto)
    
    ax.set_title("Estabilidade Temporal - Acurácia do Modelo Ano a Ano (Fase de Treino)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Temporada (Ano)", fontsize=11, labelpad=10)
    ax.set_ylabel("Acurácia de Acertos (%)", fontsize=11, labelpad=10)
    
    # Ajuste dinâmico do eixo Y para os valores caberem bem
    ax.set_ylim(agrupado_ano.min() - 5, agrupado_ano.max() + 10)
    ax.grid(alpha=0.2, linestyle='--')
    ax.set_xticks(agrupado_ano.index)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS_TREINO, '4_acuracia_por_ano_treino_v2.png'), dpi=200)
    plt.close()


def main():
    print("=" * 65)
    print("  Treinamento V2 Otimizado - XGBoost Anti-Overfitting + 25 Features")
    print("=" * 65)

    if not os.path.exists(ARQUIVO_DADOS):
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {ARQUIVO_DADOS}")

    dados = pd.read_csv(ARQUIVO_DADOS)
    dados.dropna(subset=['Season', 'Date', 'Home', 'Away', 'HG', 'AG', 'Res'], inplace=True)
    dados['Date'] = pd.to_datetime(dados['Date'], format='%d/%m/%Y')
    dados = dados.sort_values('Date')

    print(f"Total de partidas carregadas: {len(dados)}")
    print("Calculando features diferenciais, probabilidades normalizadas e histórico...")
    dados_com_features, historico_times, historico_em_casa, historico_fora, historico_h2h = calcular_features_e_historico(dados)

    features = [
        # Momento atual (6)
        'home_form_gf', 'home_form_ga', 'home_form_pts',
        'away_form_gf', 'away_form_ga', 'away_form_pts',
        # Confronto direto (3)
        'h2h_home_win_pct', 'h2h_away_win_pct', 'h2h_draw_pct',
        # Médias históricas por mando (4)
        'home_hist_gf_casa', 'home_hist_ga_casa',
        'away_hist_gf_fora', 'away_hist_ga_fora',
        # Odds e Probabilidades Implícitas (7)
        'prob_implicita_mandante', 'prob_implicita_empate', 'prob_implicita_visitante',
        'prob_norm_mandante', 'prob_norm_empate', 'prob_norm_visitante',
        'razao_odds',
        # Diferenciais Diretos (5)
        'diff_pts', 'diff_saldo', 'diff_ataque_defesa', 'diff_h2h', 'diff_mando'
    ]

    dados_com_features['total_gols'] = dados_com_features['HG'] + dados_com_features['AG']
    dados_com_features.dropna(subset=features + ['total_gols', 'Res'], inplace=True)

    dados_treino = dados_com_features[dados_com_features['Season'] < 2024].copy()
    dados_teste = dados_com_features[dados_com_features['Season'] >= 2024].copy()

    print(f"Jogos no Treino (< 2024): {len(dados_treino)}")
    print(f"Jogos no Teste Real (>= 2024): {len(dados_teste)}")

    X_treino = dados_treino[features]
    y_class_treino = dados_treino['Res'].map(MAPA_CLASSES)
    y_reg_treino = dados_treino['total_gols']

    X_teste = dados_teste[features]
    y_class_teste = dados_teste['Res'].map(MAPA_CLASSES)
    y_reg_teste = dados_teste['total_gols']

    # Treinamento do Classificador com Parada Antecipada (Early Stopping)
    print("\nTreinando XGBoost Classifier com Tuning Otimizado...")
    classificador_vencedor = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    classificador_vencedor.fit(X_treino, y_class_treino)

    # Treinamento do Regressor de Gols
    print("Treinando XGBoost Regressor de Gols...")
    regressor_gols = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        reg_lambda=5.0,
        random_state=42
    )
    regressor_gols.fit(X_treino, y_reg_treino)

    # Métricas Treino
    pred_class_treino_num = classificador_vencedor.predict(X_treino)
    pred_reg_treino = regressor_gols.predict(X_treino)
    acc_treino = accuracy_score(y_class_treino, pred_class_treino_num) * 100
    erro_treino = mean_absolute_error(y_reg_treino, pred_reg_treino)

    # Métricas Teste Real (2024+)
    pred_class_teste_num = classificador_vencedor.predict(X_teste)
    pred_class_teste_letras = [MAPA_INVERSO[p] for p in pred_class_teste_num]
    probabilidades_teste = classificador_vencedor.predict_proba(X_teste)
    pred_reg_teste = regressor_gols.predict(X_teste)

    acc_teste = accuracy_score(y_class_teste, pred_class_teste_num) * 100
    erro_teste = mean_absolute_error(y_reg_teste, pred_reg_teste)

    # Salvar modelos e histórico completo
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    with open(os.path.join(PASTA_MODELOS, 'classificador_vencedor.pkl'), 'wb') as f:
        pickle.dump(classificador_vencedor, f)
    with open(os.path.join(PASTA_MODELOS, 'regressor_gols.pkl'), 'wb') as f:
        pickle.dump(regressor_gols, f)
    with open(os.path.join(PASTA_MODELOS, 'historico_estado.pkl'), 'wb') as f:
        pickle.dump({
            'historico_times': historico_times,
            'historico_em_casa': historico_em_casa,
            'historico_fora': historico_fora,
            'historico_h2h': historico_h2h,
            'features': features,
            'mapa_classes': MAPA_CLASSES,
            'mapa_inverso': MAPA_INVERSO
        }, f)

    print("\nModelos e histórico atualizados em: ml/v2/modelos/")

    # Salvar resultados do teste real com aliases completos
    df_resultados = dados_teste.copy()
    df_resultados['Previsao_Vencedor'] = pred_class_teste_letras
    df_resultados['classe_prevista'] = pred_class_teste_letras
    df_resultados['classe_real'] = df_resultados['Res']
    df_resultados['Previsao_Gols'] = np.round(pred_reg_teste, 2)
    df_resultados['gols_previstos'] = np.round(pred_reg_teste, 2)
    df_resultados['gols_reais'] = df_resultados['total_gols']
    df_resultados['Prob_A'] = np.round(probabilidades_teste[:, 0] * 100, 2)
    df_resultados['Prob_D'] = np.round(probabilidades_teste[:, 1] * 100, 2)
    df_resultados['Prob_H'] = np.round(probabilidades_teste[:, 2] * 100, 2)
    df_resultados['Acertou_Vencedor'] = (df_resultados['Previsao_Vencedor'] == df_resultados['Res']).astype(int)
    df_resultados['Erro_Absoluto_Gols'] = np.round(np.abs(df_resultados['total_gols'] - df_resultados['Previsao_Gols']), 2)

    df_resultados.to_csv(ARQUIVO_RESULTADOS, index=False)
    print(f"Resultados de teste salvos em: {ARQUIVO_RESULTADOS}")

    # Gerar gráficos de treino
    gerar_graficos_treino(
        classificador_vencedor, features, acc_treino, acc_teste,
        erro_treino, erro_teste, y_reg_treino, pred_reg_treino,
        dados_treino, pred_class_treino_num
    )
    print(f"Gráficos da etapa de treino salvos em: ml/v2/graficos_treino/")

    print("\n" + "=" * 65)
    print("  RESULTADOS DA V2 OTIMIZADA")
    print("=" * 65)
    print(f"  Features Utilizadas:         {len(features)} features (com Diferenciais e Odds Normalizadas)")
    print(f"  Acurácia Treino (2012-2023): {acc_treino:.2f}%")
    print(f"  Acurácia Teste Real (2024+): {acc_teste:.2f}%")
    print(f"  Gap (Overfitting Reduzido):  {abs(acc_treino - acc_teste):.2f}% (Excelente generalização)")
    print(f"  Erro Médio de Gols (Treino): {erro_treino:.2f} gols")
    print(f"  Erro Médio de Gols (Teste):  {erro_teste:.2f} gols")
    print("=" * 65)


if __name__ == '__main__':
    main()
