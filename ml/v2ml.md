# ML V2 - Planejamento da Segunda Geração

> **Status:** 📝 Planejado (não iniciado)
> **Pré-requisito:** V1 deve estar treinada e funcionando (✅ concluído)

---

## 1. Objetivo

Criar uma segunda geração do modelo de Machine Learning que usa as **Odds das casas de apostas** como features adicionais, com o objetivo de superar a acurácia atual da V1.

---

## 2. Resultado da V1 (Baseline para comparar)

| Métrica | V1 |
|---|---|
| Features usadas | 13 |
| Acurácia no Teste Real (2024+) | **50.24%** |
| Erro Médio de Gols | **1.19 gols** |
| Overfitting (Treino - Teste) | 6.83% ✅ Saudável |

---

## 3. Novas Features da V2

A V2 mantém **todas as 13 features da V1** e adiciona as **Odds das casas de apostas** que já estão no `BRA.csv`.

### Features de Odds (já existem no CSV, só precisam ser usadas):

| Coluna no CSV | Descrição | Por que ajuda |
|---|---|---|
| `B365CH` | Odd da Bet365 para Vitória do Mandante | Mercado especializado já precificou a força do time |
| `B365CD` | Odd da Bet365 para Empate | Quanto maior a odd, menos provável o evento |
| `B365CA` | Odd da Bet365 para Vitória do Visitante | Diretamente relacionado à probabilidade implícita |

### Probabilidade Implícita (transformação a fazer)

Em vez de usar as odds brutas, converter para **probabilidade implícita** antes de passar para o modelo:

```
prob_implícita = 1 / odd
```

Exemplo:
- Odd Mandante = 2.0 → Probabilidade implícita = 50%
- Odd Empate = 3.5 → Probabilidade implícita = 28.6%
- Odd Visitante = 4.0 → Probabilidade implícita = 25%

Isso é mais informativo para o modelo do que a odd bruta.

### Total de Features da V2: 16

```
V1 (13):
  - home_form_gf, home_form_ga, home_form_pts
  - away_form_gf, away_form_ga, away_form_pts
  - h2h_home_win_pct, h2h_away_win_pct, h2h_draw_pct
  - home_hist_gf_casa, home_hist_ga_casa
  - away_hist_gf_fora, away_hist_ga_fora

V2 (+ 3 novas):
  - prob_implicita_mandante  (1 / B365CH)
  - prob_implicita_empate    (1 / B365CD)
  - prob_implicita_visitante (1 / B365CA)
```

---

## 4. Estrutura de Arquivos da V2



## 5. Atenção: Limitação das Odds

> ⚠️ Nem todos os jogos do CSV possuem odds preenchidas (especialmente os mais antigos, de 2012-2014).
> Na V2 o treino será feito apenas com jogos que tiverem as colunas de odds preenchidas.
> Jogos sem odds serão descartados com `dropna` nas colunas `B365CH`, `B365CD`, `B365CA`.

---

## 6. Meta de Performance da V2

| Métrica | V1 | Meta V2 |
|---|---|---|
| Acurácia Teste Real | 50.24% | **> 54%** |
| Erro Médio de Gols | 1.19 | **< 1.10** |
| Overfitting | < 10% | < 10% |

---

## 7. Quando Fazer

- [ ] API FastAPI concluída
- [ ] Frontend React concluído
- [ ] Executar `treino_v2.py`
- [ ] Comparar métricas V1 vs V2
- [ ] Atualizar API para suportar escolha de versão (`?versao=v2`)

