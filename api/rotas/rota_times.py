from fastapi import APIRouter
from api.modelos import obter_estado

roteador = APIRouter()


@roteador.get("/api/times")
def listar_times():
    """Retorna a lista de todos os times cadastrados em ordem alfabética."""
    estado = obter_estado()
    historico_times = estado['historico_times']

    # Pega os nomes dos times (chaves do dicionário) e ordena em ordem alfabética
    lista_times = sorted(list(historico_times.keys()))

    return {"times": lista_times}

