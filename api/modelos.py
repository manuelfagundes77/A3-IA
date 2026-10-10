import pickle
import pandas as pd
import os

# Caminho base para a pasta ml (relativo a onde o servidor roda)
CAMINHO_ML = os.path.join(os.path.dirname(__file__), '..', 'ml')

# Variáveis globais que guardam os modelos na memória
classificador = None
regressor = None
estado = None


def carregar_modelos():
    """Carrega os arquivos .pkl na memória uma única vez ao iniciar o servidor."""
    global classificador, regressor, estado

    caminho_classificador = os.path.join(CAMINHO_ML, 'modelos_v1', 'classificador_vencedor.pkl')
    caminho_regressor = os.path.join(CAMINHO_ML, 'modelos_v1', 'regressor_gols.pkl')
    caminho_estado = os.path.join(CAMINHO_ML, 'modelos_v1', 'historico_estado.pkl')

    with open(caminho_classificador, 'rb') as f:
        classificador = pickle.load(f)

    with open(caminho_regressor, 'rb') as f:
        regressor = pickle.load(f)

    with open(caminho_estado, 'rb') as f:
        estado = pickle.load(f)

    print(f"Modelos carregados com sucesso da pasta: {CAMINHO_ML}")


def obter_classificador():
    """Retorna o modelo classificador (quem vence)."""
    return classificador


def obter_regressor():
    """Retorna o modelo regressor (total de gols)."""
    return regressor


def obter_estado():
    """Retorna o estado completo do histórico dos times."""
    return estado


def obter_caminho_ml():
    """Retorna o caminho absoluto da pasta ml."""
    return CAMINHO_ML

