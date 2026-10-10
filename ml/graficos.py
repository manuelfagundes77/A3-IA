import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

from config_versoes import VERSAO_ATIVA, arquivo_resultados_teste, versao_esta_protegida

def main():
    # Só retreina se a versão não estiver protegida (senão usa os resultados já salvos)
    if not versao_esta_protegida(VERSAO_ATIVA):
        print("Rodando treino.py para gerar previsões atualizadas...")
        os.system("python treino.py")
    
    caminho_resultados = arquivo_resultados_teste(VERSAO_ATIVA)
    try:
        resultados = pd.read_csv(caminho_resultados)
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho_resultados} não encontrado.")
        return

    # Gráfico 1: Matriz de Confusão (acertos/erros por classe)
    print("Gerando Gráfico 1: Matriz de Confusão...")
    # Classes prováveis no Campeonato Brasileiro: H (Home), D (Draw), A (Away)
    labels = ['A', 'D', 'H']
    cm = confusion_matrix(resultados['classe_real'], resultados['classe_prevista'], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Vitória Visitante (A)', 'Empate (D)', 'Vitória Casa (H)'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Matriz de Confusão - Quem Vence (2024+)')
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Histograma de Erros (distribuição dos erros em gols)
    print("Gerando Gráfico 2: Histograma de Erros...")
    erros = resultados['gols_previstos'] - resultados['gols_reais']
    
    plt.figure(figsize=(8, 6))
    plt.hist(erros, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histograma de Erros (Gols Previstos - Gols Reais)')
    plt.xlabel('Erro de Previsão')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    # Gráfico 3: Scatter Plot (gols previsto vs real)
    print("Gerando Gráfico 3: Gols Previstos vs Reais...")
    plt.figure(figsize=(8, 6))
    
    # Adicionando um pequeno "jitter" aos gols reais para melhor visualização, já que são valores discretos
    import numpy as np
    jitter_reais = resultados['gols_reais'] + np.random.uniform(-0.1, 0.1, size=len(resultados['gols_reais']))
    
    plt.scatter(jitter_reais, resultados['gols_previstos'], alpha=0.5, color='orange')
    
    # Linha ideal de previsão
    min_val = min(resultados['gols_reais'].min(), resultados['gols_previstos'].min())
    max_val = max(resultados['gols_reais'].max(), resultados['gols_previstos'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Previsão Perfeita')
    
    plt.title('Scatter Plot - Gols Previstos vs Reais (2024+)')
    plt.xlabel('Gols Reais (com pequeno ruído para visualização)')
    plt.ylabel('Gols Previstos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

