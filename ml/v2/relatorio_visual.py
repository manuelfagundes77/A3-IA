import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configuração visual dark mode
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#0f3460'
plt.rcParams['font.family'] = 'DejaVu Sans'

PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_SAIDA = os.path.join(PASTA_ATUAL, 'graficos_teste')
ARQUIVO_RESULTADOS = os.path.join(PASTA_ATUAL, 'resultados_teste.csv')

COR_DESTAQUE = '#38bdf8'
COR_SECUNDARIA = '#0f3460'
COR_VERDE = '#4ade80'
COR_AMARELA = '#facc15'


def garantir_pasta():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA, exist_ok=True)
        print(f"Pasta '{PASTA_SAIDA}' criada.")


def carregar_dados():
    try:
        resultados = pd.read_csv(ARQUIVO_RESULTADOS)
        print(f"Dados V2 carregados: {len(resultados)} jogos no teste real (2024+).")
        return resultados
    except FileNotFoundError:
        print(f"ERRO: {ARQUIVO_RESULTADOS} não encontrado. Execute 'python treino.py' primeiro.")
        return None


def grafico_acuracia(resultados):
    print("Gerando Gráfico 1: Acurácia vs Aleatório (V2 XGBoost)...")
    acuracia_ia = (resultados['classe_real'] == resultados['classe_prevista']).mean() * 100
    acuracia_aleatoria = 100.0 / 3.0

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')

    barras = ax.bar(
        ['Chute\nAleatório', 'V2 XGBoost\n(2024+)'],
        [acuracia_aleatoria, acuracia_ia],
        color=[COR_SECUNDARIA, COR_DESTAQUE],
        width=0.4,
        edgecolor='white',
        linewidth=0.8
    )

    ax.axhline(y=acuracia_aleatoria, color=COR_AMARELA, linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline Aleatório')

    for barra, valor in zip(barras, [acuracia_aleatoria, acuracia_ia]):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.8,
                f'{valor:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold', color='white')

    ax.set_ylim(0, 80)
    ax.set_ylabel('Acurácia (%)', fontsize=13)
    ax.set_title('V2 XGBoost: Acurácia vs Chute Aleatório\n(Teste Real: Jogos de 2024+)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)

    caminho = os.path.join(PASTA_SAIDA, '1_acuracia_vs_aleatorio.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")


def grafico_matriz_confusao(resultados):
    print("Gerando Gráfico 2: Matriz de Confusão (V2 XGBoost)...")
    labels = ['A', 'D', 'H']
    nomes = ['Vitória\nVisitante', 'Empate', 'Vitória\nMandante']

    cm = confusion_matrix(resultados['classe_real'], resultados['classe_prevista'], labels=labels)
    total_por_classe = cm.sum(axis=1, keepdims=True)
    cm_pct = cm.astype(float) / total_por_classe * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    im = ax.imshow(cm_pct, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)

    for i in range(len(labels)):
        for j in range(len(labels)):
            qtd = cm[i, j]
            pct = cm_pct[i, j]
            cor_texto = 'white' if pct > 50 else '#94a3b8'
            ax.text(j, i, f'{pct:.1f}%\n({qtd} jogos)',
                    ha='center', va='center', fontsize=12, fontweight='bold', color=cor_texto)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(nomes, fontsize=11)
    ax.set_yticklabels(nomes, fontsize=11)
    ax.set_xlabel('Previsão do XGBoost', fontsize=13, labelpad=10)
    ax.set_ylabel('Resultado Real', fontsize=13, labelpad=10)
    ax.set_title('V2 XGBoost: Matriz de Confusão (% por Categoria Real)\n(Teste Real: Jogos de 2024+)', fontsize=13, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% de Acerto', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    caminho = os.path.join(PASTA_SAIDA, '2_matriz_confusao.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")


def grafico_histograma_erros(resultados):
    print("Gerando Gráfico 3: Histograma de Erros de Gols (V2)...")
    erros = resultados['gols_previstos'] - resultados['gols_reais']
    mae = np.abs(erros).mean()

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Bins de 0.5 em 0.5 perfeitamente centrados
    bins = np.arange(-5.25, 4.25, 0.5)
    n, bins, patches = ax.hist(erros, bins=bins, edgecolor='#0f3460', linewidth=1.2, rwidth=0.9)

    # Colorir as barras por faixa de acerto e exibir quantidade de partidas
    for patch, b_esq in zip(patches, bins[:-1]):
        centro = b_esq + 0.25
        altura = patch.get_height()
        if abs(centro) <= 0.5:
            patch.set_facecolor(COR_VERDE)
        elif abs(centro) <= 1.5:
            patch.set_facecolor(COR_AMARELA)
        else:
            patch.set_facecolor(COR_DESTAQUE)

        if altura > 0 and -4.0 <= centro <= 3.5:
            ax.annotate(f"{int(altura)}",
                        (centro, altura),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='white')

    ax.axvline(x=0, color='white', linestyle='--', linewidth=2, label='Erro Zero (Previsão Exata)')
    ax.axvline(x=mae, color='#f43f5e', linestyle=':', linewidth=1.8, label=f'Erro Médio Absoluto: {mae:.2f} gols')
    ax.axvline(x=-mae, color='#f43f5e', linestyle=':', linewidth=1.8)

    # Ticks de 0.5 em 0.5 com rótulos explícitos (0, 0.5, 1, 1.5, 2, etc.)
    ticks = np.arange(-4.0, 3.5 + 0.1, 0.5)
    labels = []
    for t in ticks:
        if t == 0:
            labels.append("0")
        elif t.is_integer():
            labels.append(f"{int(t)}")
        else:
            labels.append(f"{t:.1f}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlim(-4.35, 3.85)
    ax.set_ylim(0, max(n) * 1.12)

    # Linhas de grade suaves
    ax.grid(axis='y', color='#0f3460', alpha=0.5, linestyle='--')
    ax.grid(axis='x', color='#0f3460', alpha=0.2, linestyle=':')

    ax.set_xlabel('Erro na Previsão de Gols (Previsto - Real)', fontsize=12, labelpad=10)
    ax.set_ylabel('Quantidade de Partidas', fontsize=12, labelpad=10)
    ax.set_title('V2 XGBoost: Distribuição do Erro de Previsão de Gols\n(Intervalos de 0.5 gols | Teste Real: Jogos de 2024+)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left')

    caminho = os.path.join(PASTA_SAIDA, '3_histograma_erros_gols.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=180, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")


def grafico_resumo_metricas(resultados):
    print("Gerando Gráfico 4: Resumo Executivo (V2 XGBoost)...")
    acuracia = (resultados['classe_real'] == resultados['classe_prevista']).mean() * 100
    mae = np.abs(resultados['gols_previstos'] - resultados['gols_reais']).mean()
    acertos_1gol = (np.abs(resultados['gols_previstos'] - resultados['gols_reais']) <= 1.0).mean() * 100

    labels = ['A', 'D', 'H']
    cm = confusion_matrix(resultados['classe_real'], resultados['classe_prevista'], labels=labels)
    acc_por_classe = {c: (cm[i, i] / cm[i].sum() * 100) if cm[i].sum() > 0 else 0 for i, c in enumerate(labels)}

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # Titulo geral
    fig.text(0.5, 0.94, 'V2 ML-Palpites (XGBoost) - Resumo de Performance',
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.89, f'Avaliado em {len(resultados)} partidas reais do Brasileirão (2024+)',
             ha='center', va='top', fontsize=11, color='#94a3b8')

    cards = [
        {'titulo': 'Acurácia Geral', 'valor': f'{acuracia:.1f}%', 'sub': 'Supera o aleatório (33.3%)', 'cor': COR_DESTAQUE},
        {'titulo': 'Erro Médio de Gols', 'valor': f'{mae:.2f}', 'sub': 'gols de diferença em média', 'cor': COR_AMARELA},
        {'titulo': 'Precisão de Gols ±1', 'valor': f'{acertos_1gol:.1f}%', 'sub': 'das previsões com erro ≤ 1 gol', 'cor': COR_VERDE},
    ]

    for idx, card in enumerate(cards):
        ax_card = fig.add_axes([0.05 + idx * 0.31, 0.52, 0.28, 0.32])
        ax_card.set_facecolor('#16213e')
        ax_card.set_xticks([])
        ax_card.set_yticks([])
        ax_card.text(0.5, 0.78, card['titulo'], ha='center', va='center', fontsize=11, color='#94a3b8')
        ax_card.text(0.5, 0.44, card['valor'], ha='center', va='center', fontsize=26, fontweight='bold', color=card['cor'])
        ax_card.text(0.5, 0.16, card['sub'], ha='center', va='center', fontsize=9, color='white')

    ax_bar = fig.add_axes([0.05, 0.08, 0.55, 0.38])
    ax_bar.set_facecolor('#16213e')
    nomes_classes = {'H': 'Vitória Mandante', 'D': 'Empate', 'A': 'Vitória Visitante'}
    cores_classes = [COR_VERDE, COR_AMARELA, COR_DESTAQUE]
    valores = [acc_por_classe.get(c, 0) for c in ['H', 'D', 'A']]
    barras = ax_bar.bar([nomes_classes[c] for c in ['H', 'D', 'A']], valores, color=cores_classes, edgecolor='white', linewidth=0.5)
    ax_bar.axhline(y=100.0 / 3.0, color='white', linestyle='--', linewidth=1.5, alpha=0.6, label='Baseline Aleatório (33%)')
    for barra, val in zip(barras, valores):
        ax_bar.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
    ax_bar.set_ylim(0, 85)
    ax_bar.set_ylabel('Acurácia (%)', fontsize=11, color='white')
    ax_bar.set_title('Acurácia por Categoria de Resultado', fontsize=12, color='white')
    ax_bar.legend(fontsize=10)
    ax_bar.tick_params(colors='white')

    ax_txt = fig.add_axes([0.63, 0.08, 0.34, 0.38])
    ax_txt.set_facecolor(COR_SECUNDARIA)
    ax_txt.axis('off')
    texto = (
        "Sobre o Modelo V2 (XGBoost)\n\n"
        "Algoritmo:\nExtreme Gradient Boosting (500 árvores)\n\n"
        "Features Usadas (16 total):\n"
        "• 13 Features históricas da V1\n"
        "• 3 Novas de Odds Bet365 / Mercado\n"
        "  (Probabilidades Implícitas)\n\n"
        "Validação:\nDados Reais 2024+"
    )
    ax_txt.text(0.1, 0.92, texto, ha='left', va='top', fontsize=10, color='white', linespacing=1.5)

    caminho = os.path.join(PASTA_SAIDA, '4_resumo_executivo.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")


def grafico_curva_roc(resultados):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    print("Gerando Gráfico 5: Curva ROC Multiclasse (V2 XGBoost)...")
    
    classes = ['A', 'D', 'H']
    y_real = label_binarize(resultados['classe_real'], classes=classes)
    y_prob = resultados[['Prob_A', 'Prob_D', 'Prob_H']].values / 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    cores = {'A': COR_DESTAQUE, 'D': COR_AMARELA, 'H': COR_VERDE}
    nomes = {'A': 'Vitória Visitante', 'D': 'Empate', 'H': 'Vitória Mandante'}

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_real[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        # Transform FPR and TPR back to percentages for better display
        ax.plot(fpr * 100, tpr * 100, color=cores[c], lw=2.5, label=f'{nomes[c]} (Nota: {roc_auc:.2f}/1.00)')

    ax.plot([0, 100], [0, 100], color='white', lw=1.5, linestyle='--', alpha=0.5, label='Chute Aleatório (Sorte)')
    
    ax.set_xlim([0.0, 100.0])
    ax.set_ylim([0.0, 105.0])
    
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel('Taxa de ALARME FALSO (A IA disse que ia acontecer, mas errou)', fontsize=11, labelpad=10)
    ax.set_ylabel('Taxa de ACERTOS REAIS (A IA adivinhou e aconteceu mesmo)', fontsize=11, labelpad=10)
    ax.set_title('V2 XGBoost: Poder de Visão do Modelo (Curva ROC)\nRegra: Quanto mais "estufado" pro topo esquerdo, melhor e mais inteligente é a IA!', fontsize=13, fontweight='bold', pad=15)
    
    ax.legend(loc="lower right", fontsize=10, facecolor='#1a1a2e', edgecolor='white')
    ax.grid(alpha=0.2, linestyle='--')

    caminho = os.path.join(PASTA_SAIDA, '5_curva_roc.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")


def grafico_distribuicao_confianca(resultados):
    print("Gerando Gráfico 6: Distribuição de Confiança (Acertos vs Erros)...")
    
    resultados['Confianca'] = resultados[['Prob_A', 'Prob_D', 'Prob_H']].max(axis=1)
    acertos = resultados[resultados['classe_real'] == resultados['classe_prevista']]['Confianca']
    erros = resultados[resultados['classe_real'] != resultados['classe_prevista']]['Confianca']

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    pesos_acertos = np.ones(len(acertos)) / len(acertos) * 100
    pesos_erros = np.ones(len(erros)) / len(erros) * 100

    ax.hist(acertos, bins=15, alpha=0.6, color=COR_VERDE, weights=pesos_acertos, label=f'Acertos (Média: {acertos.mean():.1f}%)')
    ax.hist(erros, bins=15, alpha=0.5, color='#ef4444', weights=pesos_erros, label=f'Erros (Média: {erros.mean():.1f}%)')

    ax.axvline(x=acertos.mean(), color=COR_VERDE, linestyle='dashed', linewidth=2)
    ax.axvline(x=erros.mean(), color='#ef4444', linestyle='dashed', linewidth=2)

    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_title('V2 XGBoost: Distribuição de Confiança\nO modelo tem mais certeza quando acerta?', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Certeza do Modelo na Hora do Palpite (%)', fontsize=12)
    ax.set_ylabel('Proporção de Jogos do Grupo (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2, linestyle='--')

    caminho = os.path.join(PASTA_SAIDA, '6_distribuicao_confianca.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")


def main():
    print("=" * 60)
    print("  Gerando Relatório Visual V2 (XGBoost) para Apresentação")
    print("=" * 60)

    garantir_pasta()
    resultados = carregar_dados()
    if resultados is None:
        return

    grafico_acuracia(resultados)
    grafico_matriz_confusao(resultados)
    grafico_histograma_erros(resultados)
    grafico_resumo_metricas(resultados)
    grafico_curva_roc(resultados)
    grafico_distribuicao_confianca(resultados)

    print("\n" + "=" * 60)
    print(f"  Sucesso! 6 gráficos Dark Mode salvos em: {PASTA_SAIDA}")
    print("=" * 60)


if __name__ == '__main__':
    main()

