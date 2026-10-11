# Especificação da API (Backend) - Previsão de Futebol

Este documento define a arquitetura e os endpoints do backend que será construído com **FastAPI**. A API servirá de ponte entre o modelo de Machine Learning (`/ml`) e o Frontend em React.

## 1. Regras Gerais
- **Tecnologia:** FastAPI (Python).
- **Comunicação:** Todas as respostas serão enviadas estritamente no formato **JSON**.
- **Inicialização:** Ao ligar o servidor, a API carregará os arquivos `.pkl` na memória apenas uma vez para garantir que as respostas sejam ultrarrápidas.

---

## 2. Endpoints (Rotas)

### 2.1. Listar Times Disponíveis
- **Rota:** `GET /api/times`
- **Objetivo:** Enviar para o frontend a lista de todos os times cadastrados, para que os menus (dropdowns) de escolha sejam criados dinamicamente.
- **De onde os dados vêm:** A API vai abrir o `historico_estado.pkl`, ler as chaves do dicionário `historico_times`, ordenar em ordem alfabética e enviar.
- **Exemplo de Retorno (JSON):**
```json
{
  "times": [
    "Athletico-PR",
    "Atletico-MG",
    "Bahia",
    "Botafogo RJ",
    "Corinthians",
    "Flamengo RJ",
    "Palmeiras",
    "Sao Paulo",
    "Vasco"
  ]
}
```

### 2.2. Realizar Previsão e Raio-X
- **Rota:** `POST /api/previsao`
- **Objetivo:** O frontend envia qual é o time mandante e o visitante, e a API devolve o resultado previsto junto com todas as estatísticas para montar os gráficos no site.
- **Payload de Envio (Frontend manda):**
```json
{
  "time_casa": "Flamengo RJ",
  "time_fora": "Palmeiras"
}
```
- **Exemplo de Retorno (API devolve):**
```json
{
  "confronto": "Flamengo RJ vs Palmeiras",
  "raio_x": {
    "casa": {
      "gols_feitos_media": 2.0,
      "gols_sofridos_media": 0.4,
      "pontos_media": 3.0
    },
    "fora": {
      "gols_feitos_media": 1.4,
      "gols_sofridos_media": 0.4,
      "pontos_media": 1.8
    }
  },
  "retrospecto": {
    "vitorias_casa_pct": 40.0,
    "empates_pct": 40.0,
    "vitorias_fora_pct": 20.0
  },
  "previsao": {
    "palpite": "Vitória do(a) Flamengo RJ (Casa)",
    "chance_casa_pct": 61.5,
    "chance_empate_pct": 19.5,
    "chance_fora_pct": 19.0,
    "total_gols_esperados": 2.43,
    "placar_provavel": "2 x 1",
    "media_gols_casa": 2.15,
    "media_gols_fora": 1.05,
    "confianca": "Alta",
    "confianca_pct": 61.5
  }
}
```

> **Regra do campo `confianca`:**
> - `"Alta"` → A maior chance entre os 3 resultados for **acima de 60%** (Selo Verde no frontend)
> - `"Media"` → A maior chance estiver **entre 50% e 60%** (Selo Amarelo no frontend)
> - `"Baixa"` → A maior chance estiver **abaixo de 50%** (Selo Vermelho no frontend)

---

### 2.3. Métricas de Performance da IA
- **Rota:** `GET /api/metricas`
- **Objetivo:** Informar ao frontend a taxa de acerto real da IA para exibir em páginas do tipo "Sobre" ou em um rodapé de transparência. Ex: *"Nossa IA acerta X% dos jogos."*
- **De onde os dados vêm:** A API vai ler o arquivo `resultados_2024.csv` (gerado automaticamente pelo `treino.py`) e calcular a acurácia e o erro médio de gols em cima dos jogos reais de 2024 em diante que a IA nunca viu durante o treino.
- **Exemplo de Retorno (JSON):**
```json
{
  "periodo_avaliado": "2024 em diante",
  "total_jogos_avaliados": 380,
  "acuracia_vencedor_pct": 48.32,
  "erro_medio_gols": 1.20,
  "descricao": "A IA foi avaliada em jogos reais que nunca fizeram parte do treinamento."
}
```

