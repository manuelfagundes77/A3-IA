# Comandos Úteis - ML-Palpites

Guarde este arquivo para sempre lembrar dos comandos essenciais de navegação e inicialização do projeto no terminal.

---

## 1. Como Ligar a API (Servidor)

Abra o terminal e certifique-se de estar na **pasta raiz** do projeto (a pasta `A3 IA`). Em seguida, copie e cole o comando abaixo:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Como testar:** Assim que aparecer "Application startup complete", abra seu navegador e acesse [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
'http://127.0.0.1:8000/api/times'
- **Como desligar:** Para parar a API, clique no terminal e pressione `CTRL + C`.

---

## 2. Navegando pelo Terminal (Entrar e Voltar de Pastas)

O terminal é exigente com os caminhos. Aqui está como você se movimenta por ele:

### ⏪ Voltar para a pasta anterior
Se o seu terminal mostrar que você está dentro da pasta `ml` (ex: `C:\Users\Manuel\Desktop\A3 IA\ml>`) e você quer **voltar para a pasta principal**, use o comando:
c
```bash
cd ..
```
*(Lembrete: tem que ter um espaço entre a letra 'd' e os pontinhos).*

### ⏩ Entrar em uma pasta
Para entrar na pasta `ml` para treinar a IA, por exemplo:

```bash
cd ml
```

### 🧠 Dica de Ouro
Se você tentar rodar o comando `python treino.py` estando na pasta raiz, ele vai dar erro dizendo que não achou o arquivo. Você precisa sempre entrar na pasta certa (`cd ml`) e só depois rodar o comando!

