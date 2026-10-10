import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Configuracao visual para graficos mais bonitos e profissionais
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#0f3460'
plt.rcParams['font.family'] = 'DejaVu Sans'

PASTA_SAIDA = 'graficos_aula'
COR_DESTAQUE = '#e94560'
COR_SECUNDARIA = '#0f3460'
COR_VERDE = '#4ade80'
COR_AMARELA = '#facc15'

def garantir_pasta():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)
        print(f"Pasta '{PASTA_SAIDA}' criada.")

def carregar_dados():
    try:
        resultados = pd.read_csv('resultados_2024.csv')
        print(f"Dados carregados: {len(resultados)} jogos no teste real (2024+).")
        return resultados
    except FileNotFoundError:
        print("ERRO: Execute 'python treino.py' primeiro para gerar os dados de teste.")
        return None

# ====================================================
# GRAFICO 1: Acuracia da IA vs Chute Aleatorio
# ====================================================
def grafico_acuracia(resultados):
    print("Gerando Grafico 1: Acuracia vs Aleatorio...")
    acuracia_ia = (resultados['classe_real'] == resultados['classe_prevista']).mean() * 100
    acuracia_aleatoria = 100 / 3

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')

    barras = ax.bar(
        ['Chute\nAleatorio', 'ML-Palpites\n(2024-2026)'],
        [acuracia_aleatoria, acuracia_ia],
        color=[COR_SECUNDARIA, COR_DESTAQUE],
        width=0.4,
        edgecolor='white',
        linewidth=0.8
    )

    # Linha de referencia do aleatorio
    ax.axhline(y=acuracia_aleatoria, color=COR_AMARELA, linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline Aleatorio')

    # Valores em cima das barras
    for barra, valor in zip(barras, [acuracia_aleatoria, acuracia_ia]):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.8,
                f'{valor:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold', color='white')

    ax.set_ylim(0, 70)
    ax.set_ylabel('Acuracia (%)', fontsize=13)
    ax.set_title('Acuracia do ML-Palpites vs Chute Aleatorio\n(Teste Real: Jogos de 2024 a 2026)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)

    caminho = os.path.join(PASTA_SAIDA, '1_acuracia_vs_aleatorio.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")

# ====================================================
# GRAFICO 2: Matriz de Confusao
# ====================================================
def grafico_matriz_confusao(resultados):
    print("Gerando Grafico 2: Matriz de Confusao...")
    labels = ['A', 'D', 'H']
    nomes = ['Vitoria\nVisitante', 'Empate', 'Vitoria\nMandante']

    cm = confusion_matrix(resultados['classe_real'], resultados['classe_prevista'], labels=labels)
    total_por_classe = cm.sum(axis=1, keepdims=True)
    cm_pct = cm.astype(float) / total_por_classe * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    im = ax.imshow(cm_pct, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    for i in range(len(labels)):
        for j in range(len(labels)):
            valor_pct = cm_pct[i, j]
            valor_abs = cm[i, j]
            ax.text(j, i, f'{valor_pct:.1f}%\n({valor_abs})', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(nomes, fontsize=11)
    ax.set_yticklabels(nomes, fontsize=11)
    ax.set_xlabel('Previsto pelo ML-Palpites', fontsize=13, fontweight='bold')
    ax.set_ylabel('Resultado Real', fontsize=13, fontweight='bold')
    ax.set_title('Matriz de Confusao - Quem Vence?\n(Teste Real: 2024-2026)', fontsize=14, fontweight='bold', pad=20)

    plt.colorbar(im, ax=ax, label='% de Acerto por Classe', shrink=0.85)

    caminho = os.path.join(PASTA_SAIDA, '2_matriz_confusao.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")

# ====================================================
# GRAFICO 3: Histograma de Erros de Gols
# ====================================================
def grafico_histograma_erros(resultados):
    print("Gerando Grafico 3: Histograma de Erros de Gols...")
    erros = resultados['gols_previstos'] - resultados['gols_reais']
    erro_medio = erros.abs().mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')

    n, bins, patches = ax.hist(erros, bins=25, edgecolor='black', linewidth=0.5, color=COR_DESTAQUE, alpha=0.85)

    # Colorir barras proximas de zero de verde
    for patch, esquerda in zip(patches, bins[:-1]):
        if -1 <= esquerda <= 1:
            patch.set_facecolor(COR_VERDE)

    ax.axvline(x=0, color=COR_AMARELA, linestyle='--', linewidth=2, label='Previsao Perfeita (Erro = 0)')
    ax.axvline(x=erro_medio, color='#60a5fa', linestyle=':', linewidth=2, label=f'Erro Medio Absoluto: {erro_medio:.2f} gols')
    ax.axvline(x=-erro_medio, color='#60a5fa', linestyle=':', linewidth=2)

    ax.set_xlabel('Erro de Previsao (Gols Previstos - Gols Reais)', fontsize=13)
    ax.set_ylabel('Numero de Jogos', fontsize=13)
    ax.set_title('Distribuicao do Erro na Previsao de Gols\n(Teste Real: 2024-2026)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)

    caminho = os.path.join(PASTA_SAIDA, '3_histograma_erros_gols.png')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")

# ====================================================
# GRAFICO 4: Resumo Executivo (Card de Metricas)
# ====================================================
def grafico_resumo_metricas(resultados):
    print("Gerando Grafico 4: Resumo Executivo...")
    acuracia = (resultados['classe_real'] == resultados['classe_prevista']).mean() * 100
    erro_medio = (resultados['gols_previstos'] - resultados['gols_reais']).abs().mean()
    total_jogos = len(resultados)
    acuracia_aleatoria = 100 / 3
    ganho = acuracia - acuracia_aleatoria

    # Acuracia por tipo de resultado
    for classe, nome in [('H', 'Mandante'), ('D', 'Empate'), ('A', 'Visitante')]:
        filtro = resultados['classe_real'] == classe
        acc_classe = (resultados.loc[filtro, 'classe_real'] == resultados.loc[filtro, 'classe_prevista']).mean() * 100

    acc_por_classe = {}
    for classe in ['H', 'D', 'A']:
        filtro = resultados['classe_real'] == classe
        if filtro.sum() > 0:
            acc_por_classe[classe] = (resultados.loc[filtro, 'classe_real'] == resultados.loc[filtro, 'classe_prevista']).mean() * 100

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Relatorio de Performance - ML-Palpites\nTeste Real: Jogos de 2024 a 2026 (nunca vistos no treino)', fontsize=15, fontweight='bold', color='white', y=0.98)

    # Cards de metricas
    metricas = [
        ('Total de Jogos\nAvaliados', f'{total_jogos}', '#60a5fa'),
        ('Acuracia\nGeral', f'{acuracia:.1f}%', COR_DESTAQUE),
        ('Ganho sobre\no Aleatorio', f'+{ganho:.1f}%', COR_VERDE),
        ('Erro Medio\nem Gols', f'{erro_medio:.2f}', COR_AMARELA),
    ]

    for i, (titulo, valor, cor) in enumerate(metricas):
        ax = fig.add_axes([0.04 + i * 0.245, 0.55, 0.22, 0.35])
        ax.set_facecolor(COR_SECUNDARIA)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.65, valor, ha='center', va='center', fontsize=28, fontweight='bold', color=cor)
        ax.text(0.5, 0.2, titulo, ha='center', va='center', fontsize=11, color='#cccccc')
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)

    # Grafico de barras por tipo de resultado
    ax_bar = fig.add_axes([0.05, 0.08, 0.55, 0.38])
    ax_bar.set_facecolor('#16213e')
    nomes_classes = {'H': 'Vitoria\nMandante', 'D': 'Empate', 'A': 'Vitoria\nVisitante'}
    cores_classes = [COR_VERDE, COR_AMARELA, COR_DESTAQUE]
    valores = [acc_por_classe.get(c, 0) for c in ['H', 'D', 'A']]
    barras = ax_bar.bar([nomes_classes[c] for c in ['H', 'D', 'A']], valores, color=cores_classes, edgecolor='white', linewidth=0.5)
    ax_bar.axhline(y=acuracia_aleatoria, color='white', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Baseline Aleatorio ({acuracia_aleatoria:.0f}%)')
    for barra, val in zip(barras, valores):
        ax_bar.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
    ax_bar.set_ylim(0, 85)
    ax_bar.set_ylabel('Acuracia (%)', fontsize=11, color='white')
    ax_bar.set_title('Acuracia por Tipo de Resultado', fontsize=12, color='white')
    ax_bar.legend(fontsize=10)
    ax_bar.tick_params(colors='white')

    # Mini texto explicativo
    ax_txt = fig.add_axes([0.63, 0.08, 0.34, 0.38])
    ax_txt.set_facecolor(COR_SECUNDARIA)
    ax_txt.axis('off')
    texto = (
        "Sobre o Modelo V1\n\n"
        "Algoritmo:\nRandom Forest (200 arvores)\n\n"
        "Treinado em:\n2012 - 2023\n\n"
        "Testado em (Real):\n2024 - 2026\n\n"
        "Features usadas (13 total):\n• Momento Atual (ultimos 5j)\n• Confronto Direto (H2H)\n• Medias de Mando (Casa/Fora)"
    )
    ax_txt.text(0.1, 0.92, texto, ha='left', va='top', fontsize=10, color='white', linespacing=1.6)

    caminho = os.path.join(PASTA_SAIDA, '4_resumo_executivo.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  -> Salvo em: {caminho}")

def main():
    print("=" * 55)
    print("  Gerando Relatorio Visual para Apresentacao")
    print("=" * 55)

    garantir_pasta()
    resultados = carregar_dados()

    if resultados is None:
        return

    grafico_acuracia(resultados)
    grafico_matriz_confusao(resultados)
    grafico_histograma_erros(resultados)
    grafico_resumo_metricas(resultados)

    print("\n" + "=" * 55)
    print(f"  Pronto! 4 graficos salvos na pasta: ./{PASTA_SAIDA}/")
    print("=" * 55)

if __name__ == '__main__':
    main()

