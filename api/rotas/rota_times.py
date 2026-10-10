from fastapi import APIRouter, HTTPException
from api.modelos import obter_estado, versoes_disponiveis, VERSAO_PADRAO

roteador = APIRouter()


@roteador.get("/api/times")
def listar_times(versao: str = VERSAO_PADRAO):
    """Retorna a lista de todos os times cadastrados em ordem alfabética."""
    estado = obter_estado(versao)
    if estado is None:
        raise HTTPException(
            status_code=404,
            detail=f"Versao '{versao}' nao disponivel. Disponiveis: {versoes_disponiveis()}"
        )
    historico_times = estado['historico_times']

    # Pega os nomes dos times (chaves do dicionário) e ordena em ordem alfabética
    lista_times = sorted(list(historico_times.keys()))

    return {"versao": versao, "times": lista_times}


@roteador.get("/api/versoes")
def listar_versoes():
    """Retorna as versões de modelo carregadas e a versão padrão."""
    return {"versoes_disponiveis": versoes_disponiveis(), "versao_padrao": VERSAO_PADRAO}
