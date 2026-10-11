import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_RESULTADOS = os.path.join(PASTA_ATUAL, 'resultados_teste.csv')


def main():
    if not os.path.exists(ARQUIVO_RESULTADOS):
        print("Arquivo de resultados não encontrado. Executando treino.py da V2...")
        os.system("python treino.py")

    try:
        resultados = pd.read_csv(ARQUIVO_RESULTADOS)
    except Exception as e:
        print(f"Erro ao carregar {ARQUIVO_RESULTADOS}: {e}")
        return

    # Gráfico 1: Matriz de Confusão V2
    print("Gerando Gráfico 1: Matriz de Confusão (V2 - XGBoost)...")
    labels = ['A', 'D', 'H']
    cm = confusion_matrix(resultados['classe_real'], resultados['classe_prevista'], labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Vitória Visitante (A)', 'Empate (D)', 'Vitória Mandante (H)']
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('V2 XGBoost: Matriz de Confusão (2024+)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Histograma de Erros de Gols
    print("Gerando Gráfico 2: Histograma de Erros de Gols (V2)...")
    erros = resultados['gols_previstos'] - resultados['gols_reais']

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(-5.25, 4.25, 0.5)
    n, bins, patches = ax.hist(erros, bins=bins, color='#38BDF8', edgecolor='black', rwidth=0.9)

    for patch, b_esq in zip(patches, bins[:-1]):
        centro = b_esq + 0.25
        altura = patch.get_height()
        if altura > 0 and -4.0 <= centro <= 3.5:
            ax.annotate(f"{int(altura)}", (centro, altura), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ticks = np.arange(-4.0, 3.5 + 0.1, 0.5)
    labels = ["0" if t == 0 else (f"{int(t)}" if t.is_integer() else f"{t:.1f}") for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_xlim(-4.35, 3.85)

    plt.title('V2 XGBoost: Histograma de Erros de Gols (Intervalos de 0.5 | Previstos - Reais)', fontsize=13, fontweight='bold')
    plt.xlabel('Erro de Previsão (Gols)')
    plt.ylabel('Frequência de Jogos')
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()

    # Gráfico 3: Gols Previstos vs Reais
    print("Gerando Gráfico 3: Gols Previstos vs Reais...")
    plt.figure(figsize=(8, 6))
    jitter_reais = resultados['gols_reais'] + np.random.uniform(-0.1, 0.1, size=len(resultados['gols_reais']))
    plt.scatter(jitter_reais, resultados['gols_previstos'], alpha=0.5, color='#F59E0B')

    min_val = min(resultados['gols_reais'].min(), resultados['gols_previstos'].min())
    max_val = max(resultados['gols_reais'].max(), resultados['gols_previstos'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Previsão Perfeita')

    plt.title('V2 XGBoost: Gols Previstos vs Reais (2024+)', fontsize=13, fontweight='bold')
    plt.xlabel('Gols Reais (com leve dispersão visual)')
    plt.ylabel('Gols Previstos pelo XGBoost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico 4: Curva ROC Multiclasse
    print("Gerando Gráfico 4: Curva ROC Multiclasse...")
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    classes = ['A', 'D', 'H']
    y_real = label_binarize(resultados['classe_real'], classes=classes)
    y_prob = resultados[['Prob_A', 'Prob_D', 'Prob_H']].values / 100.0

    plt.figure(figsize=(8, 6))
    cores = {'A': '#38BDF8', 'D': '#FACC15', 'H': '#4ADE80'}
    nomes = {'A': 'Vitória Visitante', 'D': 'Empate', 'H': 'Vitória Mandante'}

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_real[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr * 100, tpr * 100, color=cores[c], lw=2, label=f'{nomes[c]} (Nota: {roc_auc:.2f}/1.00)')

    plt.plot([0, 100], [0, 100], color='gray', lw=1.5, linestyle='--', label='Chute Aleatório')
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 105.0])
    
    import matplotlib.ticker as mtick
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    
    plt.xlabel('Taxa de ALARME FALSO (A IA disse que ia acontecer, mas errou)')
    plt.ylabel('Taxa de ACERTOS REAIS (A IA adivinhou e aconteceu mesmo)')
    plt.title('V2 XGBoost: Poder de Visão do Modelo (Curva ROC)\nRegra: Quanto mais "estufado" pro topo esquerdo, melhor e mais inteligente é a IA!', fontsize=11, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 5: Distribuição de Confiança
    print("Gerando Gráfico 5: Distribuição de Confiança...")
    resultados['Confianca'] = resultados[['Prob_A', 'Prob_D', 'Prob_H']].max(axis=1)
    acertos = resultados[resultados['classe_real'] == resultados['classe_prevista']]['Confianca']
    erros = resultados[resultados['classe_real'] != resultados['classe_prevista']]['Confianca']

    plt.figure(figsize=(9, 6))
    
    pesos_acertos = np.ones(len(acertos)) / len(acertos) * 100
    pesos_erros = np.ones(len(erros)) / len(erros) * 100
    
    plt.hist(acertos, bins=15, alpha=0.6, color='#4ADE80', weights=pesos_acertos, label=f'Acertos (Média: {acertos.mean():.1f}%)')
    plt.hist(erros, bins=15, alpha=0.5, color='#EF4444', weights=pesos_erros, label=f'Erros (Média: {erros.mean():.1f}%)')

    plt.axvline(x=acertos.mean(), color='#4ADE80', linestyle='dashed', linewidth=2)
    plt.axvline(x=erros.mean(), color='#EF4444', linestyle='dashed', linewidth=2)

    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title('V2 XGBoost: Distribuição de Confiança (O modelo sabe quando vai errar?)', fontsize=13, fontweight='bold')
    plt.xlabel('Certeza do Modelo na Hora do Palpite (%)')
    plt.ylabel('Proporção de Jogos do Grupo (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()

