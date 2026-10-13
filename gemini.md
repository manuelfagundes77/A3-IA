# 🧠 Memória do Projeto: ML-Palpites (A3 IA)

> **Instrução Permanente para a IA (Gemini / Antigravity):**
> Este arquivo é a fonte da verdade e o diário de bordo do projeto. Sempre que realizar alterações estruturais, adicionar novos endpoints, atualizar o modelo de ML, criar telas no frontend ou tomar decisões arquiteturais importantes, **atualize este arquivo** mantendo o histórico de progresso sincronizado.

---

## 📌 1. Visão Geral do Projeto
- **Nome:** ML-Palpites (Sistema Preditivo de Futebol)
- **Objetivo:** Prever resultados de partidas de futebol do Campeonato Brasileiro utilizando Inteligência Artificial (Machine Learning), fornecendo:
  1. **Vencedor do jogo:** Vitória Mandante (`H`), Empate (`D`) ou Vitória Visitante (`A`), com probabilidades percentuais.
  2. **Expectativa de gols:** Previsão contínua da quantidade total de gols na partida.
  3. **Raio-X estatístico:** Métricas de momento recente (últimos 5 jogos) e retrospecto histórico direto (H2H).
- **Arquitetura Geral:**
  - 🤖 **ML (Machine Learning):** Scikit-Learn com Random Forest treinado com histórico temporal (2012-2023) e validado em dados reais (2024+).
  - ⚡ **Backend:** FastAPI (Python), servindo predições e estatísticas via endpoints REST em formato JSON.
  - 💻 **Frontend (Próxima etapa):** React SPA para visualização interativa, seleção de times e exibição de métricas.

---

## 📜 2. Regras de Ouro do Código
1. **Idioma estrito:** TODAS as variáveis, funções, classes, rotas, comentários e mensagens devem ser em **Português do Brasil (PT-BR)**.
2. **Modularidade:** Manter separadas as responsabilidades entre Machine Learning (`/ml`), Backend (`/api`) e Frontend (`/frontend`).
3. **Eficiência no Backend:** Os arquivos `.pkl` devem ser carregados para a memória RAM apenas uma vez no startup da API (`api/modelos.py`), garantindo inferência instantânea nas requisições.
4. **Git e Versionamento (GitHub Desktop):** O usuário NÃO utiliza comandos de terminal para Git, utilizando exclusivamente o **GitHub Desktop**. A IA **NUNCA** deve executar comandos `git` no terminal (como `git status`, `git add`, `git commit`, `git push`, etc.) nem sugerir fluxos de terminal para git. O versionamento é de controle manual do usuário via interface gráfica.

---

## 🗂️ 3. Estrutura de Arquivos e Funções

```text
A3 IA/
├── gemini.md                     # Memória viva do projeto para a IA (este arquivo)
├── README_IA.md                  # Contexto inicial e diretrizes gerais
├── api.md                        # Especificação técnica e contratos dos endpoints
├── COMO_RODAR.md                 # Guia prático de comandos de terminal
├── requirements.txt              # Dependências gerais do projeto (API + ML V1 + ML V2)
├── .gitignore                    # Bloqueio de subida de arquivos desnecessários pro GitHub
│
├── ml/                           # Módulo de Inteligência Artificial
│   ├── dados/
│   │   └── BRA.csv               # Dataset com histórico de jogos do Brasileirão
│   ├── treino.py                 # Pipeline de feature engineering, treino e avaliação
│   ├── predicoes.py              # Script de teste e inferência via terminal
│   ├── graficos.py               # Visualizações básicas (Matriz de confusão, erros, scatter)
│   ├── relatorio_visual.py       # Gerador de gráficos dark mode de apresentação
│   ├── ml.md                     # Planejamento e métricas da versão V1
│   ├── v2ml.md                   # Planejamento da futura versão V2 (Odds/Bet365)
│   ├── modelos_v1/               # Modelos gerados e estado salvo da V1
│   │   ├── classificador_vencedor.pkl # Modelo Random Forest Classifier treinado
│   │   ├── regressor_gols.pkl    # Modelo Random Forest Regressor treinado
│   │   ├── estatisticas_times.pkl# Estatisticas antigas
│   │   └── historico_estado.pkl  # Dicionários de histórico dos times e confrontos diretos
│   ├── graficos_treino_v1/       # Gráficos da etapa de treino/aprendizado gerados pelo treino.py
│   │   ├── 1_importancia_features.png
│   │   ├── 2_treino_vs_validacao.png
│   │   └── 3_diagnostico_gols_treino.png
│   ├── config_versoes.py         # VERSAO_ATIVA, caminhos por versão e regra de proteção (VERSOES_FINALIZADAS)
│   ├── resultados_teste_v1.csv   # Previsões vs resultados reais (teste 2024+) da V1
│   ├── graficos_teste_v1/        # Gráficos do teste real (2024+) da V1 gerados pelo relatorio_visual.py
│   └── v2/                       # Módulo autônomo da versão V2 (XGBoost com Odds)
│
└── api/                          # Módulo Backend REST (FastAPI)
    ├── main.py                   # Ponto de entrada da API, configuração de CORS e rotas
    ├── modelos.py                # Gerenciador de carregamento dos arquivos .pkl em memória
    └── rotas/
        ├── __init__.py
        ├── rota_times.py         # GET /api/times (Listagem para dropdowns)
        ├── rota_previsao.py      # POST /api/previsao (Predição, Raio-X e confiança)
        └── rota_metricas.py      # GET /api/metricas (Performance no teste real)
```

---

## ⚙️ 4. O Que Já Foi Implementado

### 4.1. Módulo de Machine Learning (V1 Concluída)
- **Dataset:** Histórico de jogos do Campeonato Brasileiro a partir de 2012 (`ml/dados/BRA.csv`).
- **Engenharia de Features (13 features no total):**
  - **Momento atual (últimos 5 jogos de cada time):** média de gols feitos (`gf`), média de gols sofridos (`ga`), média de pontos (`pts`).
  - **Confronto direto (H2H - últimos 5 jogos entre eles):** taxa de vitórias do mandante, visitante e empates.
  - **Média histórica por mando de campo:** gols feitos/sofridos pelo mandante em casa e gols feitos/sofridos pelo visitante fora.
- **Modelos:**
  - `RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)`
  - `RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)`
- **Validação temporal:**
  - Treino: Temporadas < 2024 (2012 a 2023).
  - Teste Real: Temporadas >= 2024 (dados nunca vistos pelo modelo).
- **Métricas alcançadas na V1:**
  - **Acurácia Vencedor (Treino 2012-2023):** `57.07%`
  - **Acurácia Vencedor (Teste Real 2024+):** `50.24%` (muito superior ao chute aleatório de 33.33%).
  - **Erro Médio de Gols (MAE):** `~1.14 gols (Treino)` e `~1.19 gols (Teste Real)`.
  - **Overfitting:** Saudável (apenas 6.83% de variação entre treino e teste).
- **Separação Visual dos Gráficos (Dark Mode):**
  - **Treino/Aprendizado (`ml/graficos_treino_v1/`):** gerados automaticamente ao rodar `python treino.py`. Contém: importância das 13 features (`1_importancia_features.png`), avaliação de overfitting treino vs teste (`2_treino_vs_validacao.png`) e diagnóstico de gols no treino (`3_diagnostico_gols_treino.png`).
  - **Teste Real 2024+ (`ml/graficos_teste_v1/`):** gerados por `python relatorio_visual.py`. Contém: acurácia real vs aleatório (`1_acuracia_vs_aleatorio.png`), matriz de confusão (`2_matriz_confusao.png`), histograma de erro de gols (`3_histograma_erros_gols.png`) e resumo executivo (`4_resumo_executivo.png`).
- **Versionamento e Proteção (`ml/config_versoes.py`):**
  - `VERSAO_ATIVA` define qual versão os scripts usam; todos os caminhos (`modelos_vX/`, `graficos_treino_vX/`, `graficos_teste_vX/`, `resultados_teste_vX.csv`) derivam dela.
  - `VERSOES_FINALIZADAS = []` (**V1 LIBERADA**): o professor ainda pode pedir novos gráficos/ajustes, então os arquivos da V1 **podem ser alterados se forem pedidos**. A V1 só será travada (adicionando `"v1"` à lista) quando iniciarmos a V2. Versões travadas bloqueiam `treino.py` e `relatorio_visual.py`; `predicoes.py` e `graficos.py` (somente leitura) sempre funcionam.

### 4.2. Módulo Backend REST (FastAPI Concluído)
- Servidor implementado em `api/main.py` com CORS totalmente habilitado.
- **Ciclo de vida (`startup`):** carrega na memória os `.pkl` de todas as versões existentes (`v1`, `v2`) em `api/modelos.py`. Versões sem modelos são ignoradas. Versão padrão: `v1`.
- **Rotas e Contratos:**
  1. `GET /`: Health check (`{"status": "online"}`).
  2. `GET /api/versoes`: Lista as versões carregadas e a padrão.
  3. `GET /api/times?versao=v1`: Lista ordenada em ordem alfabética com todos os clubes cadastrados em `historico_estado.pkl`.
  4. `POST /api/previsao`: Recebe `{"time_casa": "...", "time_fora": "...", "versao": "v1"}` (`versao` opcional), processa as 13 features com base no histórico e retorna:
     - Raio-X comparativo de momento atual.
     - Retrospecto de confronto direto.
     - Palpite do vencedor, probabilidades percentuais (Casa, Empate, Fora), expectativa de gols, placar provável, médias de gols (mando) e selo de confiança:
       - **Alta:** probabilidade máxima > 60%.
       - **Média:** probabilidade máxima entre 50% e 60%.
       - **Baixa:** probabilidade máxima < 50%.
       - **Campos adicionais para o mockup:** `placar_provavel` (ex: "1 x 1"), `media_gols_casa`, `media_gols_fora` e `confianca_pct`.
  5. `GET /api/metricas?versao=v1`: Lê `resultados_teste_vX.csv` e retorna dinamicamente a taxa de acerto real e o erro médio de gols.
  - Versão inexistente retorna **HTTP 404** com mensagem clara.

---

## 🚦 5. Status Atual e Roadmap

| Etapa | Módulo | Descrição | Status |
|---|---|---|---|
| 1 | ML V1 | Treinamento, 13 features e salvamento de `.pkl` | ✅ Concluído |
| 2 | Gráficos/Relatórios | Matrizes de confusão e relatórios visuais | ✅ Concluído |
| 3 | Backend API | FastAPI com rotas `/api/times`, `/api/previsao` e `/api/metricas` | ✅ Concluído |
| 4 | Frontend | Interface React consumindo a API com visual moderno | 🔄 **A Fazer** |
| 5 | ML V2 | Pasta autônoma `ml/v2/` com XGBoost (500 árvores) e 16 features (Odds Bet365/Mercado) | ✅ Concluído |

---

## 💻 6. Comandos Essenciais

### Ligar o Servidor Backend (FastAPI):
Executar a partir da pasta raiz do projeto (`A3 IA`):
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Documentação interativa Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Endpoint de times: [http://127.0.0.1:8000/api/times](http://127.0.0.1:8000/api/times)

### Retreinar o Modelo de Machine Learning:
* **Versão V1 (Random Forest):**
  ```bash
  cd ml
  python treino.py
  ```
* **Versão V2 (XGBoost com Odds):**
  ```bash
  cd ml/v2
  python treino.py
  python relatorio_visual.py
  ```

---

## 📝 7. Diário de Bordo & Registro de Alterações (Changelog)

| Data | Responsável | Ação / O que foi feito | Detalhes |
|---|---|---|---|
| **08/10/2026** | Equipe / Usuário | Finalização do ML V1 | Adicionadas 4 novas features de média histórica em casa/fora (totalizando 13 features). Acurácia subiu para 50.24% e erro médio caiu para 1.19 gols. Modelos e históricos exportados em `.pkl`. |
| **08/10/2026** | Equipe / Usuário | Construção do Backend FastAPI | Criação da arquitetura da API em `/api`: `main.py`, `modelos.py` e rotas `/api/times`, `/api/previsao`, `/api/metricas`. Suporte a CORS e carregamento eficiente na memória. |
| **08/10/2026** | Assistente (Gemini) | Criação do `gemini.md` | Criação deste documento de memória centralizada do projeto para acompanhamento e persistência das evoluções futuras. |
| **08/10/2026** | Equipe / IA | Organização de Modelos V1 | Movidos todos os arquivos `.pkl` para a nova pasta `ml/modelos_v1/`. Atualizados `treino.py`, `predicoes.py` e `api/modelos.py` para suportar o novo caminho estrutural. Preparação para a V2. |
| **08/10/2026** | Equipe / IA | Separação de Gráficos (Treino vs Teste Real) | Criada geração automática de gráficos da fase de aprendizado da V1 em `ml/graficos_treino_v1/` diretamente no `treino.py`. Mantida pasta `ml/graficos_aula/` com gráficos do Teste Real (2024+) atualizados para os 50.24% da V1. |
| **10/10/2026** | Equipe / IA | Versionamento V1/V2 e Proteção | Criado `ml/config_versoes.py` (VERSAO_ATIVA + VERSOES_FINALIZADAS). V1 congelada: `treino.py` e `relatorio_visual.py` bloqueiam sobrescrita. Renomeados `graficos_aula/` → `graficos_teste_v1/` e `resultados_2024.csv` → `resultados_teste_v1.csv`. API carrega V1/V2 e aceita parâmetro `versao`; nova rota `GET /api/versoes`. Adicionado `matplotlib` ao `requirements.txt`. |
| **10/10/2026** | Equipe / IA | Suporte ao Mockup no Backend | Expandida a rota `/api/previsao` com `placar_provavel`, `media_gols_casa`, `media_gols_fora` e `confianca_pct`, alinhando os dados da API diretamente com o design de interface planejado. |
| **10/10/2026** | Equipe / IA | Implementação e Treinamento do ML V2 (XGBoost) | Criada pasta 100% autônoma `ml/v2/`. Instalado `xgboost`, adicionadas 3 novas features de probabilidade implícita de odds (16 no total). Treinado ensemble de 500 árvores. Acurácia no teste real (2024+) subiu para 50.72%. |
| **10/10/2026** | Equipe / IA | Otimização Anti-Overfitting da V2 | Aplicadas features diferenciais (25 features no total), normalização de odds sem o vig e poda regularizada no XGBoost. Acurácia no teste real (2024+) subiu para **52.94%** e o gap de overfitting despencou de 18.28% para **3.30%** (saudável). Atualizados relatórios visuais Dark Mode. |
| **10/10/2026** | Equipe / IA | Tuning a Pedido do Usuário (V2) | Usuário solicitou relaxamento da regularização para que o modelo pudesse "aprender mais" no treino. Ajustamos hyperparameters (`max_depth=6`, `learning_rate=0.01`, `n_estimators=1000`, `gamma=1.0`) para bater ~70% de acurácia no Treino (alcançado **69.40%**). No Teste Real (2024+), a acurácia ficou em sólidos **52.07%** (muito acima do baseline V1 e do aleatório). Gráficos Dark Mode atualizados. |
| **13/10/2026** | Equipe / IA | Mockup de Arte Visual (Taste Skill) | Criado mockup em alta definição (`modelofront/mockup_taste.jpg`) elevando a referência visual do projeto (`porjetoML.jpg`) sob as diretrizes anti-slop da skill taste: eliminação de neons/blurs genéricos e adoção de identidade editorial esportiva premium. |
| **13/10/2026** | Equipe / IA | Preparação para o GitHub | Criação do `.gitignore` (bloqueando `venv`, `__pycache__` e temporários) e atualização do `api/requirements.txt` com inclusão do `xgboost` para suportar V1 e V2. |
| **13/10/2026** | Usuário / IA | Preferência por GitHub Desktop | Salva regra permanente: o usuário utiliza exclusivamente o GitHub Desktop (GUI); a IA nunca deve executar nem propor comandos `git` no terminal. |
| **13/10/2026** | Usuário / IA | Centralização do requirements.txt | Movido e organizado `requirements.txt` na raiz do projeto (cobrindo FastAPI, ML V1 e V2 com seções comentadas), facilitando instalação direta e leitura por IAs e ambientes. |

---

> 💡 **Nota para a IA:** Ao realizar qualquer nova adição (por exemplo, criação do frontend React ou implementação do ML V2 com odds), atualize a seção **4 (O Que Já Foi Implementado)**, ajuste a tabela de **Status (Seção 5)** e registre o evento na tabela do **Diário de Bordo (Seção 7)**.

