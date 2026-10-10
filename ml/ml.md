# Previsão de Futebol - Machine Learning

## 1. Criar Estrutura de Pastas

```bash
mkdir -p ml/dados
cd ml
```

Estrutura:
```
ml/
├── dados/
│   └── BRA.csv
├── treino.py
├── graficos.py
└── predicoes.py
```

---

## 2. Instalar Dependências

```bash
pip install numpy scikit-learn matplotlib
```

---

## 3. Padrão de Código

**⚠️ Importante:** Todas as variáveis, funções e comentários devem ser em **português (Brasil)**, nunca em inglês.

Exemplo:
```
dados_historico = ...
calcular_stats_time = ...
media_gols_marcados = ...
```

---

## 4. Arquivos a Criar

### `ml/treino.py`
- Ler CSV (dados/BRA.csv)
- Separar dados antes de 2024 e 2024+
- Calcular estatísticas dos times (gols marcados/sofridos, casa/fora)
- Criar features para treino
- Split 80/20 nos dados antes de 2024
- Treinar RandomForestClassifier (quem vence)
- Treinar RandomForestRegressor (total de gols)
- Salvar modelos: `classificador_vencedor.pkl` e `regressor_gols.pkl`
- Mostrar métricas: acurácia treino/teste, erro médio treino/teste
- Testar com dados de 2024+ (teste real fora da distribuição)

### `ml/graficos.py`
- Rodar `treino.py`
- Gráfico 1: Matriz de Confusão (acertos/erros por classe)
- Gráfico 2: Histograma de Erros (distribuição dos erros em gols)
- Gráfico 3: Scatter Plot (gols previsto vs real)
- Usar `plt.show()` em cada gráfico

### `ml/predicoes.py` (opcional)
- Carregar modelos (`classificador_vencedor.pkl`, `regressor_gols.pkl`)
- Função para fazer previsão com 2 times
- Mude os times manualmente e execute

---

## 5. Como Executar

**Treinar e validar:**
```bash
python treino.py
```

**Ver gráficos:**
```bash
python graficos.py
```

**Testar manualmente (opcional):**
```bash
python predicoes.py
```

---

## 6. Métricas Esperadas

- **Acurácia (quem vence):** 50-60% é aceitável
- **Erro médio (gols):** ~0.8-1.2 gols é bom
- **Overfitting:** diferença < 10% entre treino e teste

---

## 7. Como Validar o Teste Real (2024-2025)

Depois de rodar `treino.py`, você terá 2 grupos de métricas:

### Treino/Teste (80/20 em 2012-2023)
- Acurácia teste
- Erro médio teste

### Teste Real (2024-2025)
- Acurácia em 2024+
- Erro médio em 2024+

### Critérios de Sucesso

✅ **Modelo funcionando bem se:**
- Acurácia 2024+ > 50%
- Erro médio 2024+ similar ao erro de teste (diferença < 0.3)
- Diferença de acurácia entre teste e 2024+ < 15%

❌ **Modelo com problemas se:**
- Acurácia 2024+ < 40%
- Erro 2024+ >> erro teste (diferença > 0.5)
- Acurácia muito pior em 2024+ (diferença > 20% = overfitting)

---

## 8. Próximos Passos (Após ML estar pronto)

1. ✅ Treino ML
2. 🔄 API FastAPI (carrega os modelos)
3. 🔄 Frontend React

---

**Status:** 🟢 Pronto pra começar