/**
 * ============================================================================
 * ML-PALPITES | LÓGICA DO PAINEL TÉCNICO (painel_tecnico.js)
 * Consome os endpoints: /predict-rule, /forecast e /model-info da API FastAPI.
 * Todas as funções, variáveis e comentários em Português do Brasil (PT-BR).
 * ============================================================================
 */

const URL_BASE_API = "http://127.0.0.1:8000";

let versaoPainel = "v1";
let listaTimesCache = [];

document.addEventListener("DOMContentLoaded", async () => {
    lerParametrosDaUrl();
    await carregarTimesPainel();
    await executarDiagnosticoCompleto();
});

/**
 * 1. Lê os parâmetros de URL vindos da página principal (?casa=...&fora=...&versao=...)
 */
function lerParametrosDaUrl() {
    const parametros = new URLSearchParams(window.location.search);
    const versaoUrl = parametros.get("versao");
    if (versaoUrl === "v1" || versaoUrl === "v2") {
        versaoPainel = versaoUrl;
    }
    atualizarBotoesVersao();
}

/**
 * 2. Atualiza o visual dos botões V1 vs V2 no painel técnico.
 */
function atualizarBotoesVersao() {
    const btnV1 = document.getElementById("btn-painel-v1");
    const btnV2 = document.getElementById("btn-painel-v2");

    if (versaoPainel === "v1") {
        btnV1.classList.add("ativo");
        btnV2.classList.remove("ativo");
    } else {
        btnV2.classList.add("ativo");
        btnV1.classList.remove("ativo");
    }

    atualizarLinkVoltar();
}

/**
 * 3. Alterna a versão ativa e dispara a atualização completa dos 3 cards.
 */
async function selecionarVersaoPainel(novaVersao) {
    if (novaVersao === versaoPainel) return;
    versaoPainel = novaVersao;
    atualizarBotoesVersao();
    await executarDiagnosticoCompleto();
}

/**
 * 4. Busca os clubes na API (GET /api/times) e preenche os dropdowns.
 */
async function carregarTimesPainel() {
    const selectCasa = document.getElementById("select-painel-casa");
    const selectFora = document.getElementById("select-painel-fora");

    try {
        const resposta = await fetch(`${URL_BASE_API}/api/times?versao=${versaoPainel}`);
        if (!resposta.ok) throw new Error("Erro ao carregar lista de clubes");

        const dados = await resposta.json();
        listaTimesCache = dados.times || [];

        selectCasa.innerHTML = "";
        selectFora.innerHTML = "";

        listaTimesCache.forEach((time) => {
            const optCasa = document.createElement("option");
            optCasa.value = time;
            optCasa.textContent = time;
            selectCasa.appendChild(optCasa);

            const optFora = document.createElement("option");
            optFora.value = time;
            optFora.textContent = time;
            selectFora.appendChild(optFora);
        });

        // Aplica os parâmetros vindos da URL ou os padrões
        const parametros = new URLSearchParams(window.location.search);
        const casaParam = parametros.get("casa");
        const foraParam = parametros.get("fora");

        if (casaParam && listaTimesCache.includes(casaParam)) {
            selectCasa.value = casaParam;
        } else {
            selectCasa.value = listaTimesCache.includes("Palmeiras") ? "Palmeiras" : listaTimesCache[0] || "";
        }

        if (foraParam && listaTimesCache.includes(foraParam) && foraParam !== selectCasa.value) {
            selectFora.value = foraParam;
        } else {
            selectFora.value = listaTimesCache.includes("Flamengo RJ") ? "Flamengo RJ" : listaTimesCache[1] || "";
        }

        sincronizarBloqueioTimesPainel();
        atualizarLinkVoltar();
    } catch (erro) {
        console.error("Falha ao carregar clubes no painel:", erro);
    }
}

/**
 * 5. Garante que o mesmo time não seja selecionado como Mandante e Visitante.
 */
function sincronizarBloqueioTimesPainel() {
    const selectCasa = document.getElementById("select-painel-casa");
    const selectFora = document.getElementById("select-painel-fora");

    Array.from(selectCasa.options).forEach((opt) => {
        opt.disabled = opt.value === selectFora.value;
    });

    Array.from(selectFora.options).forEach((opt) => {
        opt.disabled = opt.value === selectCasa.value;
    });
}

function aoTrocarTimePainel() {
    sincronizarBloqueioTimesPainel();
    atualizarLinkVoltar();
    executarDiagnosticoCompleto();
}

function inverterTimesPainel() {
    const selectCasa = document.getElementById("select-painel-casa");
    const selectFora = document.getElementById("select-painel-fora");

    const temp = selectCasa.value;
    selectCasa.value = selectFora.value;
    selectFora.value = temp;

    sincronizarBloqueioTimesPainel();
    atualizarLinkVoltar();
    executarDiagnosticoCompleto();
}

/**
 * 6. Mantém o link "Voltar ao Simulador" sincronizado com os times selecionados.
 */
function atualizarLinkVoltar() {
    const btnVoltar = document.getElementById("btn-voltar-simulador");
    if (!btnVoltar) return;
    const selectCasa = document.getElementById("select-painel-casa");
    const selectFora = document.getElementById("select-painel-fora");
    const timeCasa = selectCasa ? encodeURIComponent(selectCasa.value) : "";
    const timeFora = selectFora ? encodeURIComponent(selectFora.value) : "";
    btnVoltar.href = `index.html?casa=${timeCasa}&fora=${timeFora}&versao=${versaoPainel}`;
}

/**
 * 7. Executa em paralelo as requisições dos 3 endpoints obrigatórios.
 */
async function executarDiagnosticoCompleto() {
    const selectCasa = document.getElementById("select-painel-casa");
    const selectFora = document.getElementById("select-painel-fora");

    const timeCasa = selectCasa.value;
    const timeFora = selectFora.value;

    if (!timeCasa || !timeFora || timeCasa === timeFora) {
        return;
    }

    document.getElementById("lbl-confronto-regra").textContent = `${timeCasa} vs ${timeFora}`;

    // Dispara as 5 chamadas simultâneas (/predict-rule, /forecast, /model-info, /explain e /cluster)
    await Promise.all([
        consultarRegraFixa(timeCasa, timeFora),
        consultarForecastGols(timeCasa, timeFora),
        consultarModelInfo(),
        consultarExplicacaoArvore(timeCasa, timeFora),
        consultarClusterTimes(timeCasa, timeFora)
    ]);
}

/**
 * 8. Consulta POST /predict-rule e atualiza o Card 1.
 */
async function consultarRegraFixa(timeCasa, timeFora) {
    const elBadgeConcordancia = document.getElementById("badge-concordancia-regra");
    const elPalpiteRegra = document.getElementById("palpite-regra-fixa");
    const elExplicacaoRegra = document.getElementById("explicacao-regra-fixa");
    const elNomeMetodoML = document.getElementById("nome-metodo-ml");
    const elAcuraciaML = document.getElementById("acuracia-ml-badge");
    const elPalpiteML = document.getElementById("palpite-ml-ia");
    const elDetalhesML = document.getElementById("detalhes-ml-ia");
    const elVantagemIA = document.getElementById("vantagem-ia-rotulo");
    const elConclusao = document.getElementById("conclusao-comparativo-texto");

    // Atualiza o link para abrir o JSON puro no navegador
    const linkExterno = document.getElementById("link-externo-predict-rule");
    if (linkExterno) {
        linkExterno.href = `${URL_BASE_API}/predict-rule?time_casa=${encodeURIComponent(timeCasa)}&time_fora=${encodeURIComponent(timeFora)}&versao=${versaoPainel}`;
    }

    try {
        const resposta = await fetch(`${URL_BASE_API}/predict-rule`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                time_casa: timeCasa,
                time_fora: timeFora,
                versao: versaoPainel
            })
        });

        if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);

        const dados = await resposta.json();

        // Status de concordância
        const saoConcordantes = dados.concordancia === "Concordam";
        elBadgeConcordancia.textContent = saoConcordantes ? "✓ Modelos Concordam" : "⚡ Modelos Discordam";
        elBadgeConcordancia.className = `badge-concordancia ${saoConcordantes ? "concordam-sim" : "concordam-nao"}`;

        // Regra Fixa
        elPalpiteRegra.textContent = dados.regra_fixa.palpite;
        elExplicacaoRegra.textContent = dados.regra_fixa.explicacao_regra;

        // Machine Learning
        const nomeModeloTexto = versaoPainel === "v2" ? "Machine Learning (V2 • XGBoost)" : "Machine Learning (V1 • Random Forest)";
        elNomeMetodoML.textContent = nomeModeloTexto;
        elAcuraciaML.textContent = `Acurácia Real: ${dados.machine_learning.acuracia_real_pct.toFixed(2)}%`;
        elPalpiteML.textContent = dados.machine_learning.palpite;

        const probs = dados.machine_learning.probabilidades_pct;
        elDetalhesML.innerHTML =
            `Probabilidades: <strong>Casa: ${probs.casa}%</strong> • <strong>Empate: ${probs.empate}%</strong> • <strong>Fora: ${probs.fora}%</strong> ` +
            `<br>Variáveis analisadas simultaneamente: <strong>${dados.machine_learning.total_features_analisadas} features</strong>`;

        elVantagemIA.innerHTML =
            `Vantagem de acurácia sobre a regra fixa: <strong>${dados.comparativo.vantagem_ml_sobre_regra_pct}</strong>`;

        elConclusao.innerHTML =
            `<strong>Diagnóstico Comparativo:</strong> ${dados.comparativo.conclusao}`;
    } catch (erro) {
        console.error("Erro ao consultar /predict-rule:", erro);
        elPalpiteRegra.textContent = "Erro na chamada";
        elPalpiteML.textContent = "Erro na chamada";
    }
}

/**
 * 9. Consulta POST /forecast e atualiza o Card 2.
 */
async function consultarForecastGols(timeCasa, timeFora) {
    const elTotalGols = document.getElementById("forecast-total-gols");
    const elPlacar = document.getElementById("forecast-placar");
    const elDivisao = document.getElementById("forecast-divisao-gols");
    const elMercado = document.getElementById("forecast-mercado");
    const elMae = document.getElementById("forecast-mae");

    // Atualiza o link para abrir o JSON puro no navegador
    const linkExterno = document.getElementById("link-externo-forecast");
    if (linkExterno) {
        linkExterno.href = `${URL_BASE_API}/forecast?time_casa=${encodeURIComponent(timeCasa)}&time_fora=${encodeURIComponent(timeFora)}&versao=${versaoPainel}`;
    }

    try {
        const resposta = await fetch(`${URL_BASE_API}/forecast`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                time_casa: timeCasa,
                time_fora: timeFora,
                versao: versaoPainel
            })
        });

        if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);

        const dados = await resposta.json();
        const prev = dados.previsao_numerica;
        elTotalGols.textContent = `${prev.total_gols_esperados.toFixed(2)} gols`;
        elPlacar.textContent = prev.placar_provavel.replace("x", "×");
        elDivisao.textContent = `${timeCasa}: ${prev.gols_esperados_casa.toFixed(2)} • ${timeFora}: ${prev.gols_esperados_fora.toFixed(2)}`;
        elMercado.textContent = prev.tendencia_mercado;
        elMae.textContent = `${prev.erro_medio_historico_mae.toFixed(2)} gols`;
    } catch (erro) {
        console.error("Erro ao consultar /forecast:", erro);
        elTotalGols.textContent = "--";
        elPlacar.textContent = "--";
    }
}

/**
 * 10. Consulta GET /model-info e atualiza o Card 3 com o catálogo de features.
 */
async function consultarModelInfo() {
    const elClassificador = document.getElementById("info-classificador");
    const elRegressor = document.getElementById("info-regressor");
    const elTotalFeatures = document.getElementById("info-total-features");
    const elAcuraciaReal = document.getElementById("info-acuracia-real");
    const elTotalJogos = document.getElementById("info-total-jogos");
    const elContador = document.getElementById("contador-features");
    const elContainerFeatures = document.getElementById("grade-features-container");

    // Atualiza o link para abrir o JSON puro no navegador
    const linkExterno = document.getElementById("link-externo-model-info");
    if (linkExterno) {
        linkExterno.href = `${URL_BASE_API}/model-info?versao=${versaoPainel}`;
    }

    try {
        const resposta = await fetch(`${URL_BASE_API}/model-info?versao=${versaoPainel}`);
        if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);

        const dados = await resposta.json();

        elClassificador.textContent = dados.algoritmo_classificador;
        elRegressor.textContent = dados.algoritmo_regressor;
        elTotalFeatures.textContent = dados.total_features;
        elAcuraciaReal.textContent = `${dados.acuracia_vencedor_pct.toFixed(2)}%`;
        elTotalJogos.textContent = `Validação em ${dados.total_jogos_avaliados} jogos (${dados.periodo_avaliado})`;
        elContador.textContent = dados.total_features;

        // Renderiza cada feature em um card limpo
        elContainerFeatures.innerHTML = "";
        dados.features_usadas.forEach((feat, index) => {
            const cardFeat = document.createElement("div");
            cardFeat.className = "item-feature-card";
            cardFeat.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="feature-codigo">${index + 1}. ${feat.nome}</span>
                </div>
                <span class="feature-desc">${feat.descricao}</span>
            `;
            elContainerFeatures.appendChild(cardFeat);
        });
    } catch (erro) {
        console.error("Erro ao consultar /model-info:", erro);
        elClassificador.textContent = "Erro na chamada";
        elRegressor.textContent = "Erro na chamada";
    }
}

/**
 * 11. Consulta POST /explain e atualiza o Card 4 (Explicação da Árvore de Decisão).
 */
async function consultarExplicacaoArvore(timeCasa, timeFora) {
    const elTipoModelo = document.getElementById("explain-tipo-modelo");
    const elPergunta = document.getElementById("explain-pergunta");
    const elDecisaoBadge = document.getElementById("explain-decisao-badge");
    const elPassosContainer = document.getElementById("explain-passos-container");
    const elFatoresContainer = document.getElementById("explain-fatores-container");
    const elResumoTexto = document.getElementById("explain-resumo-texto");

    // Atualiza o link para abrir o JSON puro no navegador
    const linkExterno = document.getElementById("link-externo-explain");
    if (linkExterno) {
        linkExterno.href = `${URL_BASE_API}/explain?time_casa=${encodeURIComponent(timeCasa)}&time_fora=${encodeURIComponent(timeFora)}&versao=${versaoPainel}`;
    }

    try {
        const resposta = await fetch(`${URL_BASE_API}/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                time_casa: timeCasa,
                time_fora: timeFora,
                versao: versaoPainel
            })
        });

        if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);

        const dados = await resposta.json();

        elTipoModelo.textContent = dados.tipo_explicador;
        elPergunta.textContent = dados.pergunta_respondida;
        elDecisaoBadge.textContent = `Decisão: ${dados.decisao_final} (${dados.confianca_decisao_pct.toFixed(1)}%)`;

        // Renderiza os 4 passos da árvore de decisão
        elPassosContainer.innerHTML = "";
        dados.caminho_arvore.forEach((passo) => {
            const cardPasso = document.createElement("div");
            cardPasso.className = "card-passo-arvore";
            cardPasso.innerHTML = `
                <div class="topo-passo-arvore">
                    <span class="badge-numero-passo">Passo ${passo.passo}</span>
                    <span class="regra-corte-arvore">${passo.regra_arvore}</span>
                </div>
                <div class="titulo-passo-arvore">${passo.variavel_nome}</div>
                <div class="texto-passo-arvore">${passo.explicacao_passo}</div>
            `;
            elPassosContainer.appendChild(cardPasso);
        });

        // Renderiza o Top 5 variáveis mais importantes na decisão
        elFatoresContainer.innerHTML = "";
        const maiorImportancia = dados.fatores_mais_importantes.length > 0
            ? dados.fatores_mais_importantes[0].importancia_no_modelo_pct
            : 100;

        dados.fatores_mais_importantes.forEach((fator, idx) => {
            const larguraRelativa = Math.max(8, Math.min(100, (fator.importancia_no_modelo_pct / maiorImportancia) * 100));
            const linha = document.createElement("div");
            linha.className = "linha-fator-explain";
            linha.innerHTML = `
                <div class="cabecalho-fator-explain">
                    <div>
                        <strong style="color: #ffffff;">${idx + 1}. ${fator.nome_legivel}</strong>
                        <span style="font-family: monospace; font-size: 0.76rem; color: #38bdf8; margin-left: 6px;">(${fator.feature} = ${fator.valor_na_partida})</span>
                    </div>
                    <strong style="color: #4ade80;">Peso: ${fator.importancia_no_modelo_pct.toFixed(2)}%</strong>
                </div>
                <div class="trilha-barra-explain">
                    <div class="preenchimento-barra-explain" style="width: ${larguraRelativa}%;"></div>
                </div>
            `;
            elFatoresContainer.appendChild(linha);
        });

        elResumoTexto.innerHTML = `<strong>Explicação Final da Árvore:</strong> ${dados.resumo_explicacao}`;
    } catch (erro) {
        console.error("Erro ao consultar /explain:", erro);
        elPergunta.textContent = "Erro ao carregar explicação da árvore.";
    }
}

/**
 * 12. Consulta GET /cluster e atualiza o Card 5 (Segmentação K-Means dos Clubes).
 */
async function consultarClusterTimes(timeCasa, timeFora) {
    const elTextoConfronto = document.getElementById("cluster-confronto-texto");
    const elBadge = document.getElementById("badge-cluster-comparativo");
    const elContainerGrupos = document.getElementById("cluster-grupos-container");

    // Atualiza o link para abrir o JSON puro no navegador
    const linkExterno = document.getElementById("link-externo-cluster");
    if (linkExterno) {
        linkExterno.href = `${URL_BASE_API}/cluster?time_casa=${encodeURIComponent(timeCasa)}&time_fora=${encodeURIComponent(timeFora)}`;
    }

    try {
        const resposta = await fetch(`${URL_BASE_API}/cluster?time_casa=${encodeURIComponent(timeCasa)}&time_fora=${encodeURIComponent(timeFora)}`);
        if (!resposta.ok) throw new Error(`HTTP ${resposta.status}`);

        const dados = await resposta.json();

        // Texto e badge do confronto atual
        elTextoConfronto.innerHTML = `
            <strong>${dados.mandante.time}</strong> (${dados.mandante.perfil}) vs <strong>${dados.visitante.time}</strong> (${dados.visitante.perfil})
            <div style="font-size: 0.83rem; color: var(--texto-secundario); margin-top: 3px;">${dados.relacao_confronto}</div>
        `;

        if (dados.mesmo_cluster) {
            elBadge.className = "badge-concordancia concordam-sim";
            elBadge.textContent = "✓ Mesmo Escalão";
        } else {
            elBadge.className = "badge-concordancia concordam-nao";
            elBadge.textContent = "⚡ Escalões Diferentes";
        }

        // Renderiza os 4 grupos com todos os times
        elContainerGrupos.innerHTML = "";
        const clusters = dados.clusters_gerais;

        Object.keys(clusters).forEach((cId) => {
            const info = clusters[cId];
            const cardGrupo = document.createElement("div");
            cardGrupo.className = `card-grupo-cluster card-cluster-${cId}`;

            const tagsHtml = info.times.map((clube) => {
                const ehTimeAtual = (clube === timeCasa || clube === timeFora);
                const classeDestaque = ehTimeAtual ? "tag-time-cluster destaque-time-atual" : "tag-time-cluster";
                const marcador = ehTimeAtual ? "★ " : "";
                return `<span class="${classeDestaque}" title="${clube}">${marcador}${clube}</span>`;
            }).join("");

            cardGrupo.innerHTML = `
                <div class="titulo-grupo-cluster">
                    <span>Cluster ${cId}: ${info.perfil.split("/")[0].trim()}</span>
                    <span style="font-size: 0.76rem; color: #38bdf8;">${info.total_times} times</span>
                </div>
                <div style="font-size: 0.77rem; color: var(--texto-suave);">
                    Média: ${info.media_pontos_cluster} pts • ${info.media_gols_cluster} gols/j
                </div>
                <div class="container-tags-times">
                    ${tagsHtml}
                </div>
            `;

            elContainerGrupos.appendChild(cardGrupo);
        });
    } catch (erro) {
        console.error("Erro ao consultar /cluster:", erro);
        if (elTextoConfronto) {
            elTextoConfronto.textContent = "Erro ao carregar dados do K-Means.";
        }
    }
}

/**
 * 13. Abre ou fecha individualmente qualquer card técnico (deixando só o título).
 */
function alternarCardTecnico(idCard) {
    const card = document.getElementById(idCard);
    if (!card) return;

    card.classList.toggle("recolhido");
    const estaRecolhido = card.classList.contains("recolhido");

    const textoBotao = card.querySelector(".texto-abre-fecha");
    if (textoBotao) {
        textoBotao.textContent = estaRecolhido ? "Abrir" : "Fechar";
    }
}

/**
 * 13. Abre ou fecha todos os cards técnicos de uma só vez.
 */
function alternarTodosOsCards(abrir) {
    const todosCards = document.querySelectorAll(".card-secao-tecnica");
    todosCards.forEach((card) => {
        if (abrir) {
            card.classList.remove("recolhido");
        } else {
            card.classList.add("recolhido");
        }
        const textoBotao = card.querySelector(".texto-abre-fecha");
        if (textoBotao) {
            textoBotao.textContent = abrir ? "Fechar" : "Abrir";
        }
    });
}

