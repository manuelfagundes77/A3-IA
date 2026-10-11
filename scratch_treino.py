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

clf = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.0,
    reg_lambda=5.0,
    reg_alpha=1.0,
    min_child_weight=3,
    random_state=42
)
clf.fit(X_treino, y_treino)
acc_tr = accuracy_score(y_treino, clf.predict(X_treino)) * 100
acc_te = accuracy_score(y_teste, clf.predict(X_teste)) * 100
print(f"TR: {acc_tr:.2f} | TE: {acc_te:.2f}")
