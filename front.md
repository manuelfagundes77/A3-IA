# 🖥️ Planejamento e Especificação do Frontend: ML-Palpites (`front.md`)

> **Referências Visuais e Técnicas:**
> - **Mockup Visual Oficial:** `modelofront/mockup_taste.jpg`
> - **Especificação Base:** `modelofront/especifica_o_de_interface_preditor_brasileir_o_ia.md`

---

## 📌 1. Visão Geral e Decisão Arquitetural
Para manter o projeto leve, direto ao ponto e sem a necessidade de instalar ferramentas extras (como Node.js, `npm` ou React), o Frontend utiliza **HTML5 + CSS3 + JavaScript Puro (Vanilla JS)** em arquivos separados dentro da pasta `frontend/`.

### Diretrizes Visuais Obrigatórias:
1. **Fidelidade ao Mockup (`modelofront/mockup_taste.jpg`):**
   - Tema escuro editorial esportivo (`#0b0f14` / `#121820`), bordas sutis (`#222b36`), tipografia limpa e alto contraste nos números.
2. **Placeholder de Escudos dos Times:**
   - Enquanto as imagens oficiais dos clubes ainda não foram adicionadas, o local do escudo exibe um **quadrado cinza** com o texto centralizado **`"Aguardando imagem"`**.
3. **Cores Padrão por Mando de Campo:**
   - **Time Mandante (Casa):** Sempre representado pela cor **VERDE** (`#129652`).
   - **Empate:** Sempre representado pela cor **ÂMBAR / AMARELO** (`#d49b35`).
   - **Time Visitante (Fora):** Sempre representado pela cor **VERMELHO** (`#c92a2a`).

---

## 🗂️ 2. Estrutura de Arquivos

```text
A3 IA/
├── front.md                  # Planejamento e especificação do Frontend (este arquivo)
└── frontend/                 # Pasta da interface web
    ├── index.html            # Estrutura semântica (Header, Seletor V1/V2, Card de Confronto e Predição)
    ├── estilo.css            # Estilos fiéis ao mockup_taste.jpg (Dark Mode Editorial Esportivo)
    └── app.js                # Funções em PT-BR consumindo a API FastAPI (V1 Random Forest e V2 XGBoost)
```

---

## 🧠 3. Seletor de Versão (V1 vs V2) e Identificação do Modelo

No topo da interface, o usuário pode alternar em tempo real entre as duas versões de Machine Learning. Um painel informativo mostra exatamente qual modelo está ativo:

| Propriedade na Tela | **Versão 1 (`v1` - Random Forest)** | **Versão 2 (`v2` - XGBoost + Odds)** |
| :--- | :--- | :--- |
| **Nome do Modelo** | `Random Forest (Classifier & Regressor)` | `XGBoost (Extreme Gradient Boosting)` |
| **Configuração Técnica** | 200 árvores (`max_depth=7`) • **13 features** | 1000 árvores (`max_depth=6`, `lr=0.01`) • **25 features** |
| **Resumo das Variáveis** | Momento (últ. 5 jogos), H2H e Mando de Campo | Base V1 + Diferenciais de Força + Odds s/ margem |
| **Acurácia Teste Real (2024+)** | **50.24%** *(via `GET /api/metricas?versao=v1`)* | **52.07%** *(via `GET /api/metricas?versao=v2`)* |
| **Erro Médio Gols (MAE)** | **1.19 gols** *(via `GET /api/metricas?versao=v1`)* | **1.17 gols** *(via `GET /api/metricas?versao=v2`)* |

---

## 🧱 4. Detalhamento dos Componentes na Tela (`index.html` + `estilo.css`)

```text
+--------------------------------------------------------------------------------------------------+
| [M] ML-PALPITES | Brasileirão Match Predictor                [ V1: Random Forest ] [ V2: XGBoost ]|
+--------------------------------------------------------------------------------------------------+
| 🧠 MODELO ATIVO: Random Forest (200 árvores • 13 features)   | Acurácia: 50.24% | MAE: 1.19 gols |
+--------------------------------------------------------------------------------------------------+

          +------------------------------------------------------------------------------+
          |  +------------------------------+          +------------------------------+  |
          |  | [ Quadrado Cinza: ]          |   (VS)   | [ Quadrado Cinza: ]          |  |
          |  | [Aguardando imagem] Time v   |   [⇄]    | [Aguardando imagem] Time v   |  |
          |  |                     Casa     |          |                     Fora     |  |
          |  +------------------------------+          +------------------------------+  |
          |                                                                              |
          |  [                      CALCULAR PALPITE COM IA                           ]  |
          +------------------------------------------------------------------------------+

          +------------------------------------------------------------------------------+
          |  PREDIÇÃO DO MODELO (IA)                           [ Confiança do Modelo ]   |
          |                                                                              |
          |  Vitória do Mandante (Casa)                                           48.5%  |
          |  [████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (Barra VERDE)   |
          |                                                                              |
          |  Empate                                                               28.0%  |
          |  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (Barra ÂMBAR)   |
          |                                                                              |
          |  Vitória do Visitante (Fora)                                          23.5%  |
          |  [█████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (Barra VERMELHA)|
          |                                                                              |
          |  +-----------------------+  +===Borda Destacada===+  +--------------------+  |
          |  | Média de Gols (Casa)  |  ‖ Placar Mais Provável‖  | Média Gols (Fora)  |  |
          |  |         1.62          |  ‖        2 × 1        ‖  |        1.05        |  |
          |  | Últ 5: GF/GA/Pts      |  ‖ Total esp: 2.43 gols‖  | Últ 5: GF/GA/Pts   |  |
          |  +-----------------------+  +=====================+  +--------------------+  |
          +------------------------------------------------------------------------------+
```

---

## 🔄 5. Funções JavaScript Planejadas (`frontend/app.js`)

Todas as funções são nomeadas e comentadas em português para fácil leitura:

1. **`inicializarAplicacao()`**: Dispara ao abrir a página, carrega a versão inicial (`v1`), preenche a ficha do modelo, busca a lista de clubes e executa a primeira análise automática.
2. **`selecionarVersaoModelo(novaVersao)`**: Alterna entre `v1` (Random Forest) e `v2` (XGBoost), atualiza a barra de ficha técnica do modelo e recalcula a partida atual.
3. **`carregarMetricasDaVersao(versao)`**: Consome `GET /api/metricas?versao={versao}` e preenche acurácia, erro médio (MAE) e total de jogos avaliados.
4. **`carregarListaDeTimes(versao)`**: Consome `GET /api/times?versao={versao}` e popula os `<select>` de Mandante e Visitante (impedindo selecionar o mesmo time nos dois lados).
5. **`inverterTimesSelecionados()`**: Botão `⇄` que inverte rapidamente o time da Casa com o time de Fora e recalcula a previsão.
6. **`calcularPalpiteComIA()`**: Consome `POST /api/previsao` enviando `{ time_casa, time_fora, versao }`.
7. **`atualizarInterfaceComPrevisao(dados)`**: Atualiza as barras horizontais (Verde = Casa, Âmbar = Empate, Vermelho = Fora), o Placar Mais Provável (`2 × 1`), as médias de gols e o Raio-X.
