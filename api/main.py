from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.modelos import carregar_modelos
from api.rotas.rota_times import roteador as roteador_times
from api.rotas.rota_previsao import roteador as roteador_previsao
from api.rotas.rota_metricas import roteador as roteador_metricas

# Criar a aplicação FastAPI
app = FastAPI(
    title="ML-Palpites API",
    description="API de previsao de resultados de futebol usando Machine Learning",
    version="1.0.0"
)

# Liberar CORS para o frontend React acessar a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em producao, trocar pelo dominio do frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar os modelos .pkl na memória e exibir as rotas ao iniciar
@app.on_event("startup")
def ao_iniciar():
    print("\n" + "=" * 50)
    print("Iniciando servidor ML-Palpites...")
    carregar_modelos()
    
    print("\nROTAS DISPONIVEIS:")
    for rota in app.routes:
        # Pega as rotas diretamente do app
        if hasattr(rota, "methods") and hasattr(rota, "path"):
            metodos = ", ".join(m for m in rota.methods if m != "OPTIONS")
            print(f"   [{metodos}] http://127.0.0.1:8000{rota.path}")
        
        # Pega as rotas dos nossos arquivos separados (roteadores incluídos)
        elif type(rota).__name__ == "_IncludedRouter" and hasattr(rota, "original_router"):
            for subrota in rota.original_router.routes:
                if hasattr(subrota, "methods") and hasattr(subrota, "path"):
                    metodos = ", ".join(m for m in subrota.methods if m != "OPTIONS")
                    print(f"   [{metodos}] http://127.0.0.1:8000{subrota.path}")

    print("\nPainel Interativo de Testes: http://127.0.0.1:8000/docs")
    print("=" * 50 + "\n")

# Registrar todas as rotas
app.include_router(roteador_times)
app.include_router(roteador_previsao)
app.include_router(roteador_metricas)

# Rota raiz (health check)
@app.get("/")
def raiz():
    return {"status": "online", "mensagem": "ML-Palpites API esta funcionando!"}

