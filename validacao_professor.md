# 📋 Validação de Requisitos do Professor (`PredictAPI`): Status de Implementação

> **Status:** ✅ **CONCLUÍDO E VALIDADO**
> **Última Atualização:** 26/10/2026
> **Escopo Implementado:** Todos os endpoints que não exigem retreinamento de modelos foram criados (`/model-info`, `/forecast`, `/predict-rule` e `/predict`), o botão **"Painel Técnico"** foi integrado na página principal e a nova tela **[`frontend/painel_tecnico.html`](frontend/painel_tecnico.html)** foi desenvolvida com botões diretos para inspecionar os JSONs das rotas.

---

## 📊 1. Checklist Geral dos Requisitos do Professor

| Requisito do Professor | Status | Onde Foi Implementado | Como Testar |
| :--- | :---: | :--- | :--- |
| **API REST sem banco** | ✅ **Concluído** | [`api/main.py`](api/main.py) e [`api/modelos.py`](api/modelos.py) | Modelos carregados em RAM no startup da API |
| **Dataset Real CSV** | ✅ **Concluído** | [`ml/dados/BRA.csv`](ml/dados/BRA.csv) | Brasileirão 2012–2024+ (1039 jogos no teste real) |
| **Classificação (`/predict`)** | ✅ **Concluído** | [`api/rotas/rota_previsao.py`](api/rotas/rota_previsao.py) | `POST /predict` ou `POST /api/previsao` |
| **Regra Fixa vs ML (`/predict-rule`)** | ✅ **Concluído** | [`api/rotas/rota_previsao.py`](api/rotas/rota_previsao.py) | `POST` ou `GET /predict-rule?time_casa=...&time_fora=...&versao=v1` |
| **Forecast / Regressão (`/forecast`)** | ✅ **Concluído** | [`api/rotas/rota_previsao.py`](api/rotas/rota_previsao.py) | `POST` ou `GET /forecast?time_casa=...&time_fora=...&versao=v1` |
| **Informações do Modelo (`/model-info`)** | ✅ **Concluído** | [`api/rotas/rota_metricas.py`](api/rotas/rota_metricas.py) | `GET /model-info?versao=v1` (ou `v2`) |
| **Árvore de Decisão (`/explain`)** | ✅ **Concluído** | [`api/rotas/rota_previsao.py`](api/rotas/rota_previsao.py) | `POST` ou `GET /explain?time_casa=...&time_fora=...&versao=v1` (ou `v2`) |
| **Frontend com Formulário e Resultado** | ✅ **Concluído** | [`frontend/index.html`](frontend/index.html) e [`frontend/painel_tecnico.html`](frontend/painel_tecnico.html) | Interface Web Dark Editorial com Cards retráteis (Abrir/Fechar) |
| **Documentação Interativa (Swagger)** | ✅ **Concluído** | FastAPI automatic docs | Acessar `http://127.0.0.1:8000/docs` |
| **Segmentação K-Means (`/cluster`)** | ✅ **Concluído** | [`ml/treino_cluster.py`](ml/treino_cluster.py) e [`api/rotas/rota_cluster.py`](api/rotas/rota_cluster.py) | `POST` ou `GET /cluster?time_casa=...&time_fora=...` + Imagem dos 38 clubes |
| **Rede Neural vs Clássico** | ⏳ *Pendente* | Planejado para benchmark | Requer treino do `MLPClassifier` para medir acurácia e ms |
| **Visão Computacional / Fala** | ⚪ *Opcional* | Não obrigatório | Definido explicitamente como bônus no documento |

---

## 🛠️ 2. O Que Foi Implementado no Backend (`api/`)

1. **`GET /model-info` (e `/api/metricas`):**
   - Retorna o tipo de algoritmo (`Random Forest` na V1 e `XGBoost` na V2).
   - Retorna a lista detalhada com nome técnico e descrição em português de todas as **13 features (V1)** ou **25 features (V2)**.
   - Retorna as métricas completas de treino (2012–2023) e do teste real (2024+).

2. **`POST` e `GET /forecast`:**
   - Previsão numérica contínua usando o modelo **`regressor_gols.pkl`**.
   - Retorna total de gols esperados, divisão dos gols por time, placar mais provável, erro médio absoluto (`MAE = 1.19 gols`) e tendência do mercado esportivo (`Over / Under 2.5`).
   - Suporte a `GET` para permitir abrir diretamente o JSON puro no navegador com um clique.

3. **`POST` e `GET /predict-rule`:**
   - Compara no mesmo confronto a decisão de uma **Regra Fixa heurística manual** (`if / else` de momento recente dos pontos + bônus de mando) contra o **Machine Learning**.
   - Mostra se os modelos **Concordaram** ou **Discordaram**.
   - Comprova a superioridade da IA: acurácia da IA (`50.24%` na V1 e `52.07%` na V2) contra a acurácia da regra fixa (`44.10%`), gerando um ganho de até `+7.97%`.
   - Suporte a `GET` para abrir o JSON puro diretamente pelo navegador.

4. **`POST /predict`:**
   - Alias oficial do endpoint `/api/previsao`, garantindo 100% de conformidade com o nome exigido no enunciado do professor.

5. **`POST` e `GET /explain`:**
   - Explica o caminho de decisões da Árvore de Decisão (`decision_path` de 4 passos) e lista as 5 variáveis de maior peso (importância %) que justificam por que o time vence, funcionando tanto na V1 quanto na V2.

6. **`POST` e `GET /cluster`:**
   - Agrupamento automático não-supervisionado via **K-Means (`n_clusters=4`)** nos dados dos 38 clubes do Brasileirão.
   - Retorna o enquadramento do confronto atual (mesmo escalão vs escalões diferentes) e a lista completa dos clubes divididos nos 4 clusters:
     - *Cluster 0:* Elite Ofensiva / Candidato ao Título (14 times)
     - *Cluster 1:* Forte Mandante / Brigando por Libertadores (11 times)
     - *Cluster 2:* Equilibrado / Meio de Tabela (10 times)
     - *Cluster 3:* Zona de Risco / Defesa Frágil (3 times)

---

## 💻 3. O Que Foi Implementado no Frontend (`frontend/`)

1. **Botão de Atalho no Cabeçalho da Página Principal ([`frontend/index.html`](frontend/index.html)):**
   - Adicionado o botão **`⚙️ Painel Técnico`** ao lado do seletor V1/V2.
   - Ao clicar nele, a aplicação passa os dois clubes e a versão ativa nos parâmetros da URL (ex: `painel_tecnico.html?casa=Palmeiras&fora=Flamengo%20RJ&versao=v1`).

2. **Segunda Tela: Painel Técnico de Avaliação ([`frontend/painel_tecnico.html`](frontend/painel_tecnico.html)):**
   - **Navegação Sincronizada:** Botão **`← Voltar ao Simulador Principal`** para retornar mantendo a mesma seleção.
   - **Cards Retráteis:** Todos os 5 cards iniciam fechados por padrão, deixando apenas o título visível com botões de alternância individual e em lote (`▼ Abrir Todos` / `▲ Fechar Todos`).
   - **Card 1 (`/predict-rule`):** Comparação visual lado a lado com badge de status (`✓ Modelos Concordam` ou `⚡ Modelos Discordam`), fórmula da regra e probabilidades do ML.
   - **Card 2 (`/forecast`):** Expectativa contínua de gols, placar provável e linha de mercado Over/Under 2.5.
   - **Card 3 (`/model-info`):** Ficha técnica dos algoritmos e grade dinâmica com o catálogo de todas as 13 ou 25 features utilizadas.
   - **Card 4 (`/explain`):** Árvore de Decisão com a pergunta respondida, caminho de 4 passos (`regra_arvore`) e barras de peso do Top 5 variáveis.
   - **Card 5 (`/cluster`):** Mapa visual de dispersão com imagem gerada de todos os 38 clubes (`grafico_clusters_kmeans.png`), centróides e catálogo completo de clubes por cluster.
   - **Botões de Inspeção Direta de JSON:** Cada um dos 5 cards possui um botão **`🔗 Abrir Rota JSON ↗`** que abre o JSON puro da rota diretamente em uma nova aba do navegador.
