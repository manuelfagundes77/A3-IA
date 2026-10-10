# Contexto do Projeto: Previsão de Futebol com IA

**Atenção IA (Instruções de Sistema)**: Leia este arquivo para entender o contexto geral do projeto antes de propor mudanças, arquiteturas ou escrever novos códigos.

## 1. Visão Geral
Este projeto é um sistema preditivo para resultados de partidas de futebol construído com Machine Learning. Utilizando o histórico de partidas (especificamente do Campeonato Brasileiro), o sistema prevê:
1. **O vencedor do jogo** (Vitória do time da Casa, Empate ou Vitória do Visitante).
2. **O número total de gols** previstos para a partida.

## 2. Regras de Ouro (Padrão de Código)
- **Idioma**: TODAS as variáveis, funções, classes, mensagens e comentários no código DEVEM ser estritamente em **Português do Brasil (PT-BR)**. Evite termos em inglês sempre que houver uma tradução adequada em programação.

## 3. Estado Atual e Estrutura
O projeto está dividido em etapas. A etapa de Inteligência Artificial / Machine Learning está **concluída** e localizada na pasta `/ml`.

**Conteúdo da pasta `/ml`**:
- `dados/BRA.csv`: Conjunto de dados histórico (dataset).
- `treino.py`: Script principal que treina algoritmos de *Random Forest* e exporta os modelos salvos (`.pkl`). A lógica usa as médias de gols marcados e sofridos como *features*.
- `predicoes.py`: Script de inferência para fazer previsões pontuais entre dois times.
- `graficos.py`: Script de análise de métricas (Matriz de confusão, histogramas, dispersão).
- Modelos gerados: `classificador_vencedor.pkl`, `regressor_gols.pkl`, `estatisticas_times.pkl`.

## 4. Roadmap e Próximos Passos
Como a etapa de ML (Machine Learning) foi validada e concluída, o desenvolvimento deve seguir o seguinte fluxo:
1. 🔄 **Backend (Em andamento/A fazer)**: Criar uma API REST utilizando **FastAPI**. A API deve carregar os arquivos `.pkl` gerados e servir rotas para prever os jogos.
2. 🔄 **Frontend (A fazer)**: Desenvolver uma interface de usuário em **React** para consumir a API do FastAPI, permitindo ao usuário final visualizar os dados e testar previsões de forma amigável.

## 5. Diretrizes de Suporte
- Se o usuário pedir para criar a API, lembre-se de importar os modelos gerados em `/ml` e expor as predições.
- Se o usuário pedir para mexer no Machine Learning, foque na pasta `/ml`.
- Mantenha sempre a modularidade (Backend e Frontend devem ser projetos/pastas isoladas do código de treinamento de ML).

