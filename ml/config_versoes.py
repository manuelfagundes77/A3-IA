"""
Configuração central de versões do ML-Palpites.

Todos os scripts da pasta ml/ (treino.py, relatorio_visual.py, predicoes.py, graficos.py)
leem os caminhos daqui, garantindo que cada versão tenha suas próprias pastas.

REGRA DE PROTEÇÃO:
    Versões listadas em VERSOES_FINALIZADAS NÃO podem ser retreinadas nem ter
    seus gráficos regerados. Isso impede que a V1 seja sobrescrita ao trabalhar na V2.
    Para liberar uma versão (só se tiver certeza!), remova-a da lista.
"""
import os

# Versão ativa usada pelos scripts. Troque para "v2" ao trabalhar na nova versão.
VERSAO_ATIVA = "v1"

# Versões congeladas (protegidas contra sobrescrita).
# ATENÇÃO: a V1 está LIBERADA de propósito. O professor ainda pode pedir novos
# gráficos/ajustes, então os arquivos da V1 podem ser alterados se forem pedidos.
# Só adicionar "v1" aqui quando formos iniciar a V2.
VERSOES_FINALIZADAS = []

# Pasta base do módulo ml (funciona independente de onde o script é chamado)
PASTA_ML = os.path.dirname(os.path.abspath(__file__))


def pasta_modelos(versao):
    """Pasta dos arquivos .pkl da versão (ex: ml/modelos_v1)."""
    return os.path.join(PASTA_ML, f"modelos_{versao}")


def pasta_graficos_treino(versao):
    """Pasta dos gráficos da fase de treino/aprendizado (ex: ml/graficos_treino_v1)."""
    return os.path.join(PASTA_ML, f"graficos_treino_{versao}")


def pasta_graficos_teste(versao):
    """Pasta dos gráficos do teste real 2024+ (ex: ml/graficos_teste_v1)."""
    return os.path.join(PASTA_ML, f"graficos_teste_{versao}")


def arquivo_resultados_teste(versao):
    """CSV com previsões vs resultados reais do teste (ex: ml/resultados_teste_v1.csv)."""
    return os.path.join(PASTA_ML, f"resultados_teste_{versao}.csv")


def versao_esta_protegida(versao):
    """Retorna True se a versão está finalizada e não pode ser sobrescrita."""
    return versao in VERSOES_FINALIZADAS


def bloquear_se_protegida(versao, acao):
    """Interrompe o script com uma mensagem clara caso a versão esteja protegida."""
    if versao_esta_protegida(versao):
        print("=" * 60)
        print(f"  BLOQUEADO: a versao '{versao}' esta FINALIZADA e protegida.")
        print(f"  A acao '{acao}' iria sobrescrever os arquivos dela.")
        print(f"  -> Para trabalhar em outra versao, altere VERSAO_ATIVA em config_versoes.py")
        print(f"  -> Para liberar a '{versao}', remova-a de VERSOES_FINALIZADAS (cuidado!)")
        print("=" * 60)
        return True
    return False

