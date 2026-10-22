import sys
sys.path.append('.')
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
from ml.v2.treino import calcular_features_e_historico, MAPA_CLASSES

dados = pd.read_csv('ml/dados/BRA.csv')
dados.dropna(subset=['Season', 'Date', 'Home', 'Away', 'HG', 'AG', 'Res'], inplace=True)
dados['Date'] = pd.to_datetime(dados['Date'], format='%d/%m/%Y')
dados = dados.sort_values('Date')

dados_com_features, _, _, _, _ = calcular_features_e_historico(dados)
features = ['home_form_gf', 'home_form_ga', 'home_form_pts', 'away_form_gf', 'away_form_ga', 'away_form_pts', 'h2h_home_win_pct', 'h2h_away_win_pct', 'h2h_draw_pct', 'home_hist_gf_casa', 'home_hist_ga_casa', 'away_hist_gf_fora', 'away_hist_ga_fora', 'prob_implicita_mandante', 'prob_implicita_empate', 'prob_implicita_visitante', 'prob_norm_mandante', 'prob_norm_empate', 'prob_norm_visitante', 'razao_odds', 'diff_pts', 'diff_saldo', 'diff_ataque_defesa', 'diff_h2h', 'diff_mando']
dados_com_features['total_gols'] = dados_com_features['HG'] + dados_com_features['AG']
dados_com_features.dropna(subset=features + ['total_gols', 'Res'], inplace=True)

X_treino = dados_com_features[dados_com_features['Season'] < 2024][features]
y_treino = dados_com_features[dados_com_features['Season'] < 2024]['Res'].map(MAPA_CLASSES)
X_teste = dados_com_features[dados_com_features['Season'] >= 2024][features]
y_teste = dados_com_features[dados_com_features['Season'] >= 2024]['Res'].map(MAPA_CLASSES)

# ==============================================================================
# HIPERPARÂMETROS DO MODELO (Altere aqui para testar diferentes combinações):
# ==============================================================================
clf = xgb.XGBClassifier(
    # 🌲 1. QUANTIDADE DE ÁRVORES (n_estimators)
    # Quantas árvores de decisão em sequência o modelo vai criar.
    # -> Aumentar (ex: 1500, 2000): Dá mais tempo para aprender, mas pode demorar e decorar demais.
    # -> Diminuir (ex: 300, 500): Treina muito rápido, mas pode aprender de menos.
    n_estimators=1000,

    # 📏 2. PROFUNDIDADE MÁXIMA DA ÁRVORE (max_depth)
    # Quantos níveis de perguntas ("se... então...") cada árvore pode fazer.
    # -> Aumentar (ex: 7, 8, 10): Aprende detalhes e regras muito específicas (sobe o treino, risco no teste).
    # -> Diminuir (ex: 3, 4, 5): Árvores mais simples, gerais e conservadoras.
    max_depth=6,

    # 🐢 3. TAXA DE APRENDIZADO (learning_rate)
    # A velocidade/tamanho do passo com que cada nova árvore corrige os erros da anterior.
    # -> Aumentar (ex: 0.05, 0.1): Aprende rápido com poucas árvores, mas pode errar o ponto ideal.
    # -> Diminuir (ex: 0.005, 0.01): Aprendizado bem cuidadoso e suave (exige mais n_estimators).
    learning_rate=0.01,

    # 🎲 4. AMOSTRAGEM DE JOGOS (subsample)
    # Porcentagem de jogos sorteados para treinar cada árvore individual (0.8 = 80%).
    # -> Evita que uma única árvore veja todos os jogos e fique "viciada".
    subsample=0.8,

    # 📊 5. AMOSTRAGEM DE FEATURES/COLUNAS (colsample_bytree)
    # Porcentagem de estatísticas/features sorteadas para cada árvore (0.8 = 80%).
    # -> Força o modelo a não depender sempre das mesmas features (como só as odds).
    colsample_bytree=0.8,

    # ✂️ 6. RIGOR PARA CRIAR NOVOS RAMOS (gamma)
    # O ganho mínimo de certeza exigido para a árvore se dividir em um novo galho.
    # -> Aumentar (ex: 2.0, 5.0): Modelo mais rígido/conservador, corta divisões fracas.
    # -> Diminuir (ex: 0.0, 0.5): Modelo mais livre para criar galhos.
    gamma=1.0,

    # 🛡️ 7. REGULARIZAÇÃO L2 (reg_lambda)
    # Penalidade suave nos pesos das decisões (estilo Ridge).
    # -> Aumentar (ex: 8.0, 10.0): Puxa as previsões extremas para o centro, reduzindo o risco de erro feio.
    reg_lambda=5.0,

    # 🧹 8. REGULARIZAÇÃO L1 (reg_alpha)
    # Penalidade que pode zerar a importância de regras fracas (estilo Lasso).
    # -> Aumentar (ex: 2.0, 5.0): Descarta conexões fracas, limpando o modelo.
    reg_alpha=1.0,

    # ⚖️ 9. PESO MÍNIMO POR FOLHA (min_child_weight)
    # Quantidade mínima de dados/jogos que precisam cair em uma ponta para validar a regra.
    # -> Aumentar (ex: 5, 8): Impede que o modelo crie uma regra baseada em apenas 1 ou 2 jogos raros.
    min_child_weight=3,

    # 🔒 10. SEMENTE ALEATÓRIA (random_state)
    # Garante que sempre que rodar com os mesmos números, o resultado seja idêntico.
    random_state=42
)

print("\nTreinando o modelo de teste rápido...")
clf.fit(X_treino, y_treino)

acc_tr = accuracy_score(y_treino, clf.predict(X_treino)) * 100
acc_te = accuracy_score(y_teste, clf.predict(X_teste)) * 100

print("\n" + "=" * 60)
print(">>> RESULTADO DO TESTE RAPIDO (XGBoost)")
print("=" * 60)
print(f"[*] Treino (2012-2023):    {acc_tr:.2f}%  -> O quanto o modelo memorizou")
print(f"[*] Teste Real (2024+):    {acc_te:.2f}%  -> O que ele acerta em jogos novos")
print(f"[*] Diferenca (Overfit):   {acc_tr - acc_te:.2f}%  -> Ideal manter abaixo de ~18%")
print("=" * 60 + "\n")
