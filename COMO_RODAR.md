# 🚀 Como Rodar o Projeto (ML-Palpites)

Todos os comandos abaixo foram feitos para serem rodados **diretamente na pasta raiz do projeto (`A3 IA`)**, sem precisar ficar entrando e saindo de pastas no terminal.

---

## 1. Ligar a API (Servidor Backend)

Abra o terminal na pasta raiz (`A3 IA`) e rode:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Documentação da API (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Para desligar a API:** clique no terminal e aperte `CTRL + C`.

---

## 2. Abrir o Frontend (HTML no Navegador)

Com a API ligada no passo 1, você tem duas formas simples de abrir o sistema:

- **Pelo Windows (Mais fácil):** Abra a pasta `frontend` e dê **dois cliques** no arquivo `index.html`.
- **Pelo Terminal (na pasta raiz):**
  ```bash
  start frontend/index.html
  ```

---

## 3. Rodar Novamente o Treino da IA (V1 e V2)

Se quiser retreinar os modelos e atualizar os gráficos, basta rodar os comandos abaixo direto da pasta raiz (`A3 IA`):

### 🔹 Treinar a Versão 1 (Random Forest - 13 Features)
```bash
python ml/treino.py
```
*(Para atualizar também os gráficos de teste real da V1, rode: `python ml/relatorio_visual.py`)*

### 🔸 Treinar a Versão 2 (XGBoost com Odds - 25 Features)
```bash
python ml/v2/treino.py
```
*(Para atualizar também os gráficos de teste real da V2, rode: `python ml/v2/relatorio_visual.py`)*

### 📊 Atualizar o Agrupamento K-Means (Clusters)
```bash
python ml/treino_cluster.py
```

---

## 4. Instalar Dependências (Apenas na 1ª vez ou em outro PC)

Se baixou o projeto em um computador novo, instale todas as bibliotecas de uma vez rodando na pasta raiz:

```bash
pip install -r requirements.txt
```
