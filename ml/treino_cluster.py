import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Caminhos de arquivos
PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_ATUAL, 'dados', 'BRA.csv')
ARQUIVO_PKL = os.path.join(PASTA_ATUAL, 'kmeans_times.pkl')
PASTA_FRONTEND = os.path.join(os.path.dirname(PASTA_ATUAL), 'frontend')
ARQUIVO_IMAGEM = os.path.join(PASTA_FRONTEND, 'grafico_clusters_kmeans.png')


def calcular_estatisticas_clubes(df):
    """Calcula métricas agregadas por clube para alimentação do K-Means."""
    clubes = sorted(pd.concat([df['Home'], df['Away']]).unique())
    dados_clubes = []

    for clube in clubes:
        jogos_casa = df[df['Home'] == clube]
        jogos_fora = df[df['Away'] == clube]

        total_jogos = len(jogos_casa) + len(jogos_fora)
        if total_jogos < 19:
            continue

        gols_pro = jogos_casa['HG'].sum() + jogos_fora['AG'].sum()
        gols_contra = jogos_casa['AG'].sum() + jogos_fora['HG'].sum()

        vitorias_casa = (jogos_casa['Res'] == 'H').sum()
        empates_casa = (jogos_casa['Res'] == 'D').sum()
        pontos_casa = (vitorias_casa * 3) + empates_casa

        vitorias_fora = (jogos_fora['Res'] == 'A').sum()
        empates_fora = (jogos_fora['Res'] == 'D').sum()
        pontos_fora = (vitorias_fora * 3) + empates_fora

        pontos_total = pontos_casa + pontos_fora

        media_pontos = pontos_total / total_jogos
        media_gols_pro = gols_pro / total_jogos
        media_gols_contra = gols_contra / total_jogos
        aprov_casa = pontos_casa / len(jogos_casa) if len(jogos_casa) > 0 else 0.0
        aprov_fora = pontos_fora / len(jogos_fora) if len(jogos_fora) > 0 else 0.0

        dados_clubes.append({
            'clube': clube,
            'total_jogos': total_jogos,
            'media_pontos': round(media_pontos, 3),
            'media_gols_pro': round(media_gols_pro, 3),
            'media_gols_contra': round(media_gols_contra, 3),
            'aprov_casa': round(aprov_casa, 3),
            'aprov_fora': round(aprov_fora, 3)
        })

    return pd.DataFrame(dados_clubes)


def gerar_grafico_clusters(df_agrupado, centros, labels_ordenados):
    """Gera visualização em Dark Mode com todos os clubes agrupados."""
    os.makedirs(os.path.dirname(ARQUIVO_IMAGEM), exist_ok=True)

    cor_fundo = '#0a0e13'
    cor_card = '#111822'
    cor_borda = '#233245'
    cor_texto = '#f3f6fa'
    cor_sec = '#9ba6b5'

    plt.rcParams.update({
        'figure.facecolor': cor_fundo,
        'axes.facecolor': cor_card,
        'axes.edgecolor': cor_borda,
        'axes.labelcolor': cor_texto,
        'xtick.color': cor_sec,
        'ytick.color': cor_sec,
        'text.color': cor_texto,
        'grid.color': cor_borda,
        'grid.alpha': 0.35
    })

    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)

    cores_clusters = {
        0: '#10b981',  # Elite - Verde esmeralda
        1: '#38bdf8',  # Forte Mandante - Azul céu
        2: '#f59e0b',  # Meio de Tabela - Âmbar
        3: '#ef4444'   # Zona de Risco - Vermelho
    }

    nomes_perfis = {
        0: 'Cluster 0: Elite Ofensiva / Título',
        1: 'Cluster 1: Forte Mandante / G6',
        2: 'Cluster 2: Equilibrado / Meio de Tabela',
        3: 'Cluster 3: Zona de Alerta / Defesa Frágil'
    }

    # Plotar cada cluster
    for c_id in range(4):
        subset = df_agrupado[df_agrupado['cluster_ordenado'] == c_id]
        ax.scatter(
            subset['media_gols_pro'],
            subset['media_pontos'],
            c=cores_clusters[c_id],
            s=120,
            alpha=0.9,
            edgecolors='#ffffff',
            linewidth=1.2,
            label=f"{nomes_perfis[c_id]} ({len(subset)} times)",
            zorder=4
        )

        # Plotar nomes dos clubes
        for _, row in subset.iterrows():
            ax.annotate(
                row['clube'],
                (row['media_gols_pro'], row['media_pontos']),
                xytext=(0, 7),
                textcoords='offset points',
                fontsize=8.2,
                fontweight='600',
                color=cor_texto,
                ha='center',
                va='bottom',
                alpha=0.95
            )

    # Plotar centróides (estrelas)
    for c_id in range(4):
        centro_x = centros[c_id][1]  # media_gols_pro
        centro_y = centros[c_id][0]  # media_pontos
        ax.scatter(
            centro_x, centro_y,
            c='#ffffff',
            marker='*',
            s=340,
            edgecolors=cores_clusters[c_id],
            linewidth=2,
            zorder=6
        )

    ax.set_title(
        'Segmentação K-Means dos Clubes do Brasileirão (Modern Era: 2012–2024+)\n'
        'Agrupamento Automático Baseado em Aproveitamento, Gols Feitos e Força de Mando',
        fontsize=13,
        fontweight='bold',
        pad=18
    )
    ax.set_xlabel('Média de Gols Pró por Partida (Poder Ofensivo)', fontsize=11, labelpad=10)
    ax.set_ylabel('Média de Pontos por Partida (Aproveitamento)', fontsize=11, labelpad=10)
    ax.grid(True, linestyle='--')
    ax.legend(
        facecolor=cor_card,
        edgecolor=cor_borda,
        fontsize=9,
        loc='upper left',
        framealpha=0.95
    )

    plt.tight_layout()
    plt.savefig(ARQUIVO_IMAGEM, dpi=180, facecolor=cor_fundo)
    plt.close()
    print(f"Gráfico de clusters salvo com sucesso em: {ARQUIVO_IMAGEM}")


def main():
    print("=" * 60)
    print("  Treinamento do Módulo K-Means (Segmentação de Clubes)")
    print("=" * 60)

    if not os.path.exists(ARQUIVO_DADOS):
        raise FileNotFoundError(f"Arquivo {ARQUIVO_DADOS} não encontrado.")

    df = pd.read_csv(ARQUIVO_DADOS)
    df_clubes = calcular_estatisticas_clubes(df)
    print(f"Total de clubes qualificados para agrupamento: {len(df_clubes)}")

    features_cluster = ['media_pontos', 'media_gols_pro', 'media_gols_contra', 'aprov_casa', 'aprov_fora']
    X = df_clubes[features_cluster].values

    # Treinamento do K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    labels_brutos = kmeans.fit_predict(X)
    df_clubes['cluster_bruto'] = labels_brutos

    # Ordenar os clusters para que 0 seja sempre a Elite e 3 seja a Zona de Risco
    medias_pontos_por_cluster = df_clubes.groupby('cluster_bruto')['media_pontos'].mean().sort_values(ascending=False)
    mapa_ordenacao = {antigo: novo for novo, antigo in enumerate(medias_pontos_por_cluster.index)}
    df_clubes['cluster_ordenado'] = df_clubes['cluster_bruto'].map(mapa_ordenacao)

    nomes_perfis = {
        0: 'Elite Ofensiva / Candidato ao Título',
        1: 'Forte Mandante / Brigando por Libertadores',
        2: 'Equilibrado / Meio de Tabela',
        3: 'Zona de Risco / Defesa Frágil'
    }

    df_clubes['perfil'] = df_clubes['cluster_ordenado'].map(nomes_perfis)

    # Centróides reordenados
    centros_reordenados = [kmeans.cluster_centers_[antigo] for antigo in medias_pontos_por_cluster.index]

    # Gerar a imagem visual
    gerar_grafico_clusters(df_clubes, centros_reordenados, nomes_perfis)

    # Dicionário serializável para exportar no .pkl
    dados_serializaveis = {
        'times': {},
        'clusters': {}
    }

    for c_id in range(4):
        times_do_cluster = df_clubes[df_clubes['cluster_ordenado'] == c_id].sort_values('media_pontos', ascending=False)
        dados_serializaveis['clusters'][c_id] = {
            'cluster': c_id,
            'perfil': nomes_perfis[c_id],
            'total_times': len(times_do_cluster),
            'media_pontos_cluster': round(float(times_do_cluster['media_pontos'].mean()), 2),
            'media_gols_cluster': round(float(times_do_cluster['media_gols_pro'].mean()), 2),
            'times': times_do_cluster['clube'].tolist()
        }

    for _, row in df_clubes.iterrows():
        dados_serializaveis['times'][row['clube']] = {
            'time': row['clube'],
            'cluster': int(row['cluster_ordenado']),
            'perfil': row['perfil'],
            'total_jogos': int(row['total_jogos']),
            'media_pontos': float(row['media_pontos']),
            'media_gols_pro': float(row['media_gols_pro']),
            'media_gols_contra': float(row['media_gols_contra']),
            'aprov_casa': float(row['aprov_casa']),
            'aprov_fora': float(row['aprov_fora'])
        }

    with open(ARQUIVO_PKL, 'wb') as f:
        pickle.dump(dados_serializaveis, f)

    print(f"Modelo e perfis salvos em: {ARQUIVO_PKL}")
    print("\nResumo dos Clusters:")
    for c_id, info in dados_serializaveis['clusters'].items():
        print(f"  Cluster {c_id} ({info['perfil']}): {info['total_times']} times -> {info['times'][:4]}...")
    print("=" * 60)


if __name__ == '__main__':
    main()

