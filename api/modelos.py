import pickle
import pandas as pd
import os

# Caminho base para a pasta ml (relativo a onde o servidor roda)
CAMINHO_ML = os.path.join(os.path.dirname(__file__), '..', 'ml')

# Versões que a API tenta carregar no startup e a versão padrão
VERSOES_SUPORTADAS = ["v1", "v2"]
VERSAO_PADRAO = "v1"

# Cache global: {"v1": {"classificador": ..., "regressor": ..., "estado": ...}, ...}
modelos_por_versao = {}


def carregar_modelos():
    """Carrega os arquivos .pkl de todas as versões disponíveis na memória, uma única vez."""
    for versao in VERSOES_SUPORTADAS:
        pasta = os.path.join(CAMINHO_ML, f'modelos_{versao}')
        try:
            with open(os.path.join(pasta, 'classificador_vencedor.pkl'), 'rb') as f:
                classificador = pickle.load(f)
            with open(os.path.join(pasta, 'regressor_gols.pkl'), 'rb') as f:
                regressor = pickle.load(f)
            with open(os.path.join(pasta, 'historico_estado.pkl'), 'rb') as f:
                estado = pickle.load(f)
        except FileNotFoundError:
            print(f"Versao {versao}: modelos nao encontrados (ignorada).")
            continue

        modelos_por_versao[versao] = {
            "classificador": classificador,
            "regressor": regressor,
            "estado": estado,
        }
        print(f"Versao {versao}: modelos carregados de {pasta}")


def versoes_disponiveis():
    """Lista as versões que foram carregadas com sucesso."""
    return sorted(modelos_por_versao.keys())


def obter_modelos(versao=VERSAO_PADRAO):
    """Retorna o dicionário de modelos da versão, ou None se ela não estiver disponível."""
    return modelos_por_versao.get(versao)


def obter_classificador(versao=VERSAO_PADRAO):
    """Retorna o modelo classificador (quem vence)."""
    modelos = obter_modelos(versao)
    return modelos["classificador"] if modelos else None


def obter_regressor(versao=VERSAO_PADRAO):
    """Retorna o modelo regressor (total de gols)."""
    modelos = obter_modelos(versao)
    return modelos["regressor"] if modelos else None


def obter_estado(versao=VERSAO_PADRAO):
    """Retorna o estado completo do histórico dos times."""
    modelos = obter_modelos(versao)
    return modelos["estado"] if modelos else None


def obter_caminho_ml():
    """Retorna o caminho absoluto da pasta ml."""
    return CAMINHO_ML


def obter_caminho_resultados(versao=VERSAO_PADRAO):
    """Retorna o caminho do CSV de resultados do teste real da versão."""
    return os.path.join(CAMINHO_ML, f'resultados_teste_{versao}.csv')
