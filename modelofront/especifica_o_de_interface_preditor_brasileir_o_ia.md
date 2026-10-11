# Especificação Técnica de Interface: Simulador Preditivo de Futebol IA (Brasileirão)

Este documento descreve detalhadamente a arquitetura de interface, os componentes visuais, o fluxo de dados e a integração frontend com a API de Machine Learning para predições de partidas do Campeonato Brasileiro Série A.
imagem de refecia esta na pasta modelofront  porjetiML.jpg

---

## 1. Visão Geral da Interface

* **Tema e Identidade Visual:** Dark Theme esportivo/analítico (tons de grafite escuro `zinc-950` a `zinc-900`, com elementos em `zinc-800`), alto contraste para números e métricas, e acentos de cor vibrantes baseados nos clubes.
* **Proposta do Painel:** Centralizar a escolha dinâmica de mandante e visitante em um card superior e exibir, logo abaixo, a predição da IA em barras horizontais coloridas com as probabilidades e o raio-x de gols.
* **Responsividade:** Layout adaptável em coluna única no mobile e visualização em grade dividida em tablets e desktops.

---

## 2. Contrato e Mapeamento de Endpoints da API

A aplicação consome três rotas principais:

### 2.1. Listar Times (`GET /api/times`)
* **Momento da chamada:** No carregamento inicial da página (`componentDidMount` / `useEffect`).
* **Objetivo:** Popular os seletores suspensos (dropdowns) com todos os clubes que o modelo de ML conhece.
* **Exemplo de Retorno:**
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

### 2.2. Realizar Previsão e Raio-X (`POST /api/previsao`)
* **Momento da chamada:** Ao clicar no botão **"CALCULAR PALPITE COM IA"**.
* **Payload Enviado:**
```json
{
  "time_casa": "Flamengo RJ",
  "time_fora": "Palmeiras"
}
```
* **Exemplo de Retorno:**
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
    "confianca": "Alta"
  }
}
```

### 2.3. Métricas de Performance do Modelo (`GET /api/metricas`)
* **Momento da chamada:** No carregamento inicial da página.
* **Objetivo:** Alimentar a barra de transparência/confiabilidade no rodapé.
* **Exemplo de Retorno:**
```json
{
  "periodo_avaliado": "2024 em diante",
  "total_jogos_avaliados": 380,
  "acuracia_vencedor_pct": 48.32,
  "erro_medio_gols": 1.20,
  "descricao": "A IA foi avaliada em jogos reais que nunca fizeram parte do treinamento."
}
```

---

## 3. Mapeamento de Metadados no Frontend (Escudos e Cores)

Como a rota `/api/times` retorna apenas os nomes dos times em texto puro, o frontend mantém um dicionário de metadados para injetar o escudo oficial e a cor predominante de cada equipe:

```typescript
export interface TimeMeta {
  escudo: string;
  corPredominante: string;
  nomeExibicao: string;
}

export const METADADOS_TIMES: Record<string, TimeMeta> = {
  "Flamengo RJ": {
    nomeExibicao: "Flamengo",
    escudo: "/assets/escudos/flamengo.svg",
    corPredominante: "#C22A1E" // Vermelho Rubro-Negro
  },
  "Palmeiras": {
    nomeExibicao: "Palmeiras",
    escudo: "/assets/escudos/palmeiras.svg",
    corPredominante: "#006437" // Verde Alviverde
  },
  "Botafogo RJ": {
    nomeExibicao: "Botafogo",
    escudo: "/assets/escudos/botafogo.svg",
    corPredominante: "#202020" // Preto/Grafite
  },
  "Corinthians": {
    nomeExibicao: "Corinthians",
    escudo: "/assets/escudos/corinthians.svg",
    corPredominante: "#18181B" // Preto e Branco
  },
  "Atletico-MG": {
    nomeExibicao: "Atlético-MG",
    escudo: "/assets/escudos/atletico-mg.svg",
    corPredominante: "#27272A" // Preto Alvinegro
  },
  "Sao Paulo": {
    nomeExibicao: "São Paulo",
    escudo: "/assets/escudos/sao-paulo.svg",
    corPredominante: "#DC2626" // Tricolor / Vermelho
  },
  "Bahia": {
    nomeExibicao: "Bahia",
    escudo: "/assets/escudos/bahia.svg",
    corPredominante: "#0284C7" // Azul Tricolor
  },
  "Athletico-PR": {
    nomeExibicao: "Athletico-PR",
    escudo: "/assets/escudos/athletico-pr.svg",
    corPredominante: "#B91C1C" // Vermelho Furacão
  },
  "Vasco": {
    nomeExibicao: "Vasco",
    escudo: "/assets/escudos/vasco.svg",
    corPredominante: "#1E293B" // Faixa Alvinegra
  }
};
```

---

## 4. Detalhamento dos Componentes de Interface

```
+----------------------------------------------------------------------------------+
|                             SELETOR DE CONFRONTO                                 |
|  [ CASA (MANDANTE) ]                 [ VS ]              [ FORA (VISITANTE) ]    |
|  (Escudo) [ Dropdown Flamengo v ]   ( <-> )   (Escudo) [ Dropdown Palmeiras v ]  |
+----------------------------------------------------------------------------------+
|                         [ BOTÃO: CALCULAR PALPITE COM IA ]                       |
+----------------------------------------------------------------------------------+
|                                                                                  |
| PREDIÇÃO DO MODELO (IA)                                  [ SELO CONFIANÇA: ALTA ]|
| Palpite: Vitória do(a) Flamengo RJ (Casa)                                        |
|                                                                                  |
| [Escudo] Vitória do Flamengo RJ (Mandante)  ====================>         61.5%  |
| [Ícone]  Empate                             =======>                      19.5%  |
| [Escudo] Vitória do Palmeiras (Visitante)   ======>                       19.0%  |
|                                                                                  |
| +-------------------------+ +-------------------------+ +----------------------+ |
| | Média Gols Flamengo     | | Total Gols Esperados    | | Média Gols Palmeiras | |
| |          2.00           | |          2.43           | |         1.40         | |
| | Sofridos: 0.4 | Pts: 3.0| | Over/Under: 2.5 gols    | | Sofridos: 0.4| Pts: 1.8|
| +-------------------------+ +-------------------------+ +----------------------+ |
|                                                                                  |
+----------------------------------------------------------------------------------+
| [RODAPÉ] IA avaliada em 380 jogos reais (2024+) • Acurácia: 48.32% • Erro: 1.20 |
+----------------------------------------------------------------------------------+
```

### 4.1. Bloco Superior: Seleção de Times

1. **Card Mandante (Esquerda):**
   * **Tag Superior:** `CASA (MANDANTE)` com estilo sutil e cor neutra/esmeralda.
   * **Escudo Dinâmico:** Círculo com borda destacada exibindo o escudo associado ao time selecionado.
   * **Dropdown Dinâmico:** Preenchido pela rota `/api/times`.
   * **Regra de Validação:** Desabilita o time que já estiver selecionado no dropdown de visitante.

2. **Divisor Central (`VS` e Inversão):**
   * Texto estilizado `VS` no centro vertical.
   * **Botão Inverter ($\rightleftarrows$):** Ao clicar, troca instantaneamente os valores de `time_casa` e `time_fora`, invertendo também os escudos exibidos.

3. **Card Visitante (Direita):**
   * **Tag Superior:** `FORA (VISITANTE)`.
   * **Escudo Dinâmico:** Exibe o escudo oficial do clube visitante.
   * **Dropdown Dinâmico:** Lista os mesmos clubes, bloqueando o mandante ativo.

4. **Botão de Execução:**
   * **Texto:** `CALCULAR PALPITE COM IA`.
   * **Estados:**
     * *Normal:* Fundo verde vibrante (`bg-emerald-500`), texto branco em negrito.
     * *Carregando:* Exibe spinner circular e texto *"Calculando probabilidades..."*.
     * *Desabilitado:* Caso os dois campos não estejam definidos.

---

### 4.2. Bloco Central: Painel de Predições

Revelado logo abaixo após a conclusão da requisição `POST /api/previsao`.

#### A. Cabeçalho e Selo de Confiança
* **Título:** `PREDIÇÃO DO MODELO (IA)`.
* **Destaque do Palpite:** Subtítulo com texto em destaque exibindo o campo `previsao.palpite`.
* **Badge de Confiança Semântico (`previsao.confianca`):**
  * `"Alta"`: Fundo `emerald-950/70`, borda `emerald-500/40`, texto `emerald-400` com ponto luminoso verde.
  * `"Media"`: Fundo `amber-950/70`, borda `amber-500/40`, texto `amber-400` com ponto luminoso amarelo.
  * `"Baixa"`: Fundo `rose-950/70`, borda `rose-500/40`, texto `rose-400` com ponto luminoso vermelho.

#### B. Gráfico de Barras Horizontais
Três barras empilhadas com fundo de trilha escuro (`bg-zinc-950`), bordas suaves e preenchimento animado:

1. **Barra 1 — Vitória do Mandante:**
   * **Ícone:** Miniatura do escudo do mandante à esquerda.
   * **Rótulo:** `Vitória do {time_casa} (Mandante)`.
   * **Cor de Fundo:** `METADADOS_TIMES[time_casa].corPredominante`.
   * **Largura:** `{previsao.chance_casa_pct}%`.
   * **Valor:** `{previsao.chance_casa_pct}%` alinhado à direita com tipografia em negrito.

2. **Barra 2 — Empate:**
   * **Ícone:** Ícone neutro de empate / divisão.
   * **Rótulo:** `Empate`.
   * **Cor de Fundo:** Âmbar / Dourado neutro (`#EAB308`).
   * **Largura:** `{previsao.chance_empate_pct}%`.
   * **Valor:** `{previsao.chance_empate_pct}%` alinhado à direita.

3. **Barra 3 — Vitória do Visitante:**
   * **Ícone:** Miniatura do escudo do visitante à esquerda.
   * **Rótulo:** `Vitória do {time_fora} (Visitante)`.
   * **Cor de Fundo:** `METADADOS_TIMES[time_fora].corPredominante`.
   * **Largura:** `{previsao.chance_fora_pct}%`.
   * **Valor:** `{previsao.chance_fora_pct}%` alinhado à direita.

---

### 4.3. Bloco Inferior: Cards de Raio-X e Médias de Gols

Grid de 3 colunas horizontais exibindo métricas complementares:

1. **Card Esquerdo — Gols Mandante:**
   * **Título:** `Média de Gols ({time_casa})`.
   * **Valor Central:** `{raio_x.casa.gols_feitos_media.toFixed(2)}` (ex: `2.00`).
   * **Linha Informativa:** `Gols sofridos: {raio_x.casa.gols_sofridos_media} • Média pts: {raio_x.casa.pontos_media}`.

2. **Card Central — Expectativa Total da Partida:**
   * **Título:** `Total de Gols Esperados`.
   * **Valor Central:** `{previsao.total_gols_esperados.toFixed(2)}` (ex: `2.43`).
   * **Linha Informativa:** Indicação da linha de mercado (ex: *"Linha esperada: Mais de 2.5 gols"* se $\ge 2.5$, senão *"Menos de 2.5 gols"*).

3. **Card Direito — Gols Visitante:**
   * **Título:** `Média de Gols ({time_fora})`.
   * **Valor Central:** `{raio_x.fora.gols_feitos_media.toFixed(2)}` (ex: `1.40`).
   * **Linha Informativa:** `Gols sofridos: {raio_x.fora.gols_sofridos_media} • Média pts: {raio_x.fora.pontos_media}`.

---

### 4.4. Bloco de Rodapé: Transparência e Performance da IA

Barra horizontal discreta e elegante posicionada no rodapé do painel, populada com o retorno de `GET /api/metricas`:

* **Texto Exibido:**
  > *"Modelo de Machine Learning avaliado em {total_jogos_avaliados} partidas reais ({periodo_avaliado}) • Acurácia de vencedor: {acuracia_vencedor_pct}% • Erro médio de gols: {erro_medio_gols}"*
* **Estilo:** Tipografia reduzida (`text-xs`), cor suave (`text-zinc-500`), borda superior fina (`border-t border-zinc-800`).

---

## 5. Regras de Comportamento e UX (User Experience)

1. **Animação de Transição:** As barras horizontais devem utilizar animação CSS (`transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1)`) para preencher da esquerda para a direita toda vez que uma nova predição for executada.
2. **Preenchimento Inicial Padrão:** Ao abrir a página e receber os times, selecionar automaticamente o primeiro item como Mandante e o segundo como Visitante para evitar telas em branco.
3. **Persistência de Dados de Retrospecto:** Caso deseje exibir futuramente o retrospecto histórico retornado em `retrospecto`, pode-se adicionar abas comutáveis: `[Previsão da IA]` e `[Histórico Direto]`.