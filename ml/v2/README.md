# Módulo ML V2: XGBoost com Features Diferenciais e Odds Normalizadas

Este módulo contém a **Segunda Geração Otimizada (V2)** do sistema preditivo do Brasileirão, desenvolvida de forma 100% isolada dentro da pasta `ml/v2/`.

---

## 🚀 Melhorias Implementadas na V2

1. **Combate Rigoroso ao Overfitting:**
   - Redução da profundidade máxima para `max_depth=4` com poda de nós (`gamma=1.5`).
   - Regularização L2 pesada (`reg_lambda=7.0`) e regularização L1 (`reg_alpha=1.0`).
   - Taxa de aprendizado suave (`learning_rate=0.015`) com 250 árvores especializadas.
   - **Resultado:** O gap de overfitting caiu de **18.28%** para apenas **3.30%**.

2. **Engenharia de 25 Features:**
   - **13 Features Históricas da V1:** Momento recente (gf, ga, pts), Confronto direto (H2H) e médias históricas de mando.
   - **7 Features de Mercado e Odds:** Probabilidades implícitas brutas, probabilidades normalizadas sem o "vig" (margem da casa) e razão de odds de favoritismo.
   - **5 Features Diferenciais Diretas:** Diferencial de pontos recentes (`diff_pts`), saldo de gols (`diff_saldo`), choque ataque vs defesa (`diff_ataque_defesa`), saldo histórico H2H (`diff_h2h`) e diferencial histórico de mando (`diff_mando`).

---

## 📊 Performance Comparativa

| Métrica | V1 (Random Forest) | V2 Inicial | **V2 Otimizada (Atual)** |
|---|---|---|---|
| **Features** | 13 | 16 | **25** |
| **Acurácia Treino** | 57.07% | 69.01% | **56.24%** |
| **Acurácia Teste Real (2024+)** | 50.24% | 50.72% | **52.94%** 🔥 |
| **Overfitting (Gap Treino-Teste)** | 6.83% | 18.29% ⚠️ | **3.30%** ✅ |
| **Erro Médio de Gols (Teste)** | 1.19 gols | 1.19 gols | **1.18 gols** |

---

## 📁 Estrutura de Arquivos

```text
ml/v2/
├── treino.py               # Pipeline de treino XGBoost com regularização e 25 features
├── predicoes.py            # Simulador de partidas interativo
├── graficos.py             # Visualização de matriz de confusão e erros
├── relatorio_visual.py     # Gerador de relatórios visuais Dark Mode
├── README.md               # Esta documentação
│
├── modelos/                # Modelos treinados salvos em formato .pkl
│   ├── classificador_vencedor.pkl
│   ├── regressor_gols.pkl
│   └── historico_estado.pkl
│
├── graficos_treino/        # Gráficos da fase de treinamento
│   ├── 1_importancia_features_v2.png
│   ├── 2_treino_vs_validacao_v2.png
│   └── 3_diagnostico_gols_treino_v2.png
│
├── graficos_teste/         # Gráficos de validação no teste real (2024+)
│   ├── 1_acuracia_vs_aleatorio.png
│   ├── 2_matriz_confusao.png
│   ├── 3_histograma_erros_gols.png
│   └── 4_resumo_executivo.png
│
└── resultados_teste.csv    # 1.039 jogos de 2024+ com previsões vs realidade
```

---

## 🛠️ Como Executar

```bash
cd ml/v2
python treino.py
python relatorio_visual.py
python predicoes.py
```
