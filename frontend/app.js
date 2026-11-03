/**
 * ============================================================================
 * ML-PALPITES | LÓGICA DO FRONTEND (app.js)
 * Comunicação direta via fetch() com a API FastAPI (Python)
 * Todas as funções e variáveis estão em Português do Brasil (PT-BR).
 * ============================================================================
 */

const URL_BASE_API = "http://127.0.0.1:8000";

// Estado global da interface
let versaoAtual = "v1";
let listaDeTimesCache = [];
let metricasAtuaisCache = {
    v1: { acuraciaTreino: 57.07, acuraciaReal: 50.24, ganhoSorte: 16.91, erroGols: 1.19, totalJogos: 1039 },
    v2: { acuraciaTreino: 69.40, acuraciaReal: 52.07, ganhoSorte: 18.74, erroGols: 1.19, totalJogos: 1039 }
};

// Dicionário completo da Ficha Técnica dos modelos de Machine Learning (V1 vs V2)
const DESCRICOES_MODELOS = {
    v1: {
        badge: "MODELO V1 ATIVO",
        algoritmo: "Random Forest (Classifier & Regressor)",
        parametros: "200 árvores (max_depth=7, random_state=42) • 13 features históricas",
        explicacao: "Engenharia de Features (13): Momento recente nos últimos 5 jogos (média de gols feitos, sofridos e pontos), Confronto Direto (H2H) e Média Histórica de Mando de Campo (Casa/Fora)."
    },
    v2: {
        badge: "MODELO V2 ATIVO",
        algoritmo: "XGBoost (Extreme Gradient Boosting)",
        parametros: "1000 árvores (max_depth=6, lr=0.01, gamma=1.0) • 25 features + Odds",
        explicacao: "Engenharia de Features (25): Base histórica da V1 (13) + Diferenciais Diretos de Força (Saldo, Pontos, Ataque vs Defesa, H2H e Mando) + Probabilidades Implícitas de Odds sem margem (Vig)."
    }
};

/**
 * 1. Inicializa a aplicação assim que o HTML terminar de carregar.
 */
document.addEventListener("DOMContentLoaded", () => {
    inicializarAplicacao();
});

async function inicializarAplicacao() {
    // Lê parâmetros da URL caso o usuário esteja voltando do Painel Técnico
    const parametros = new URLSearchParams(window.location.search);
    const versaoUrl = parametros.get("versao");
    if (versaoUrl === "v1" || versaoUrl === "v2") {
        versaoAtual = versaoUrl;
        const btnV1 = document.getElementById("btn-versao-v1");
        const btnV2 = document.getElementById("btn-versao-v2");
        if (versaoAtual === "v2") {
            btnV2.classList.add("ativo");
            btnV1.classList.remove("ativo");
        } else {
            btnV1.classList.add("ativo");
            btnV2.classList.remove("ativo");
        }
    }

    atualizarFichaTecnicaDoModelo(versaoAtual);
    await carregarMetricasDaVersao(versaoAtual);
    await carregarListaDeTimes(versaoAtual);
}

/**
 * 2. Alterna entre a Versão 1 (Random Forest) e a Versão 2 (XGBoost).
 */
async function selecionarVersaoModelo(novaVersao) {
    if (novaVersao === versaoAtual) return;

    versaoAtual = novaVersao;

    // Atualiza o visual dos botões no topo
    const btnV1 = document.getElementById("btn-versao-v1");
    const btnV2 = document.getElementById("btn-versao-v2");

    if (versaoAtual === "v1") {
        btnV1.classList.add("ativo");
        btnV2.classList.remove("ativo");
    } else {
        btnV2.classList.add("ativo");
        btnV1.classList.remove("ativo");
    }

    // Atualiza os textos explicativos da ficha técnica do modelo
    atualizarFichaTecnicaDoModelo(versaoAtual);

    // Busca as métricas reais da versão escolhida na API
    await carregarMetricasDaVersao(versaoAtual);

    // Recalcula automaticamente o confronto atual com o novo modelo selecionado
    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");
    atualizarLinkPainelTecnico();
    if (selectCasa.value && selectFora.value && selectCasa.value !== selectFora.value) {
        await calcularPalpiteComIA();
    }
}

/**
 * 3. Atualiza os textos da faixa superior que explica qual IA está rodando.
 */
function atualizarFichaTecnicaDoModelo(versao) {
    const info = DESCRICOES_MODELOS[versao] || DESCRICOES_MODELOS.v1;
    const met = metricasAtuaisCache[versao] || metricasAtuaisCache.v1;

    document.getElementById("ficha-badge-versao").textContent = info.badge;
    document.getElementById("ficha-nome-algoritmo").textContent = info.algoritmo;
    document.getElementById("ficha-parametros").textContent = info.parametros;
    document.getElementById("ficha-explicacao-features").innerHTML =
        `<strong>${info.explicacao.split(":")[0]}:</strong>${info.explicacao.substring(info.explicacao.indexOf(":") + 1)}`;

    // Atualiza imediatamente com os valores de referência enquanto a API responde
    document.getElementById("ficha-acuracia-treino").textContent = `${met.acuraciaTreino.toFixed(2)}%`;
    document.getElementById("ficha-acuracia").textContent = `${met.acuraciaReal.toFixed(2)}%`;
    document.getElementById("ficha-ganho-sorte").textContent = `+${met.ganhoSorte.toFixed(2)}%`;
    document.getElementById("ficha-erro-gols").textContent = `${met.erroGols} gols`;
    document.getElementById("ficha-total-jogos").textContent = `${met.totalJogos} jogos`;

    // Atualiza também o selo de acurácia no card de baixo
    const valorConfianca = document.getElementById("valor-confianca-ia");
    if (valorConfianca) {
        valorConfianca.textContent = `${met.acuraciaReal.toFixed(2)}% (Treino: ${met.acuraciaTreino.toFixed(2)}%)`;
    }
}

/**
 * 4. Busca na API (GET /api/metricas) a acurácia e o erro médio de gols no Teste Real (2024+).
 */
async function carregarMetricasDaVersao(versao) {
    const elAcuraciaTreino = document.getElementById("ficha-acuracia-treino");
    const elAcuracia = document.getElementById("ficha-acuracia");
    const elGanhoSorte = document.getElementById("ficha-ganho-sorte");
    const elErroGols = document.getElementById("ficha-erro-gols");
    const elTotalJogos = document.getElementById("ficha-total-jogos");
    const elRodape = document.getElementById("rodape-metricas-ia");
    const valorConfianca = document.getElementById("valor-confianca-ia");

    try {
        const resposta = await fetch(`${URL_BASE_API}/api/metricas?versao=${versao}`);
        if (!resposta.ok) {
            throw new Error(`Erro ao buscar métricas da versão ${versao}`);
        }

        const dados = await resposta.json();
        const nomeModeloCurto = versao === "v2" ? "XGBoost (V2)" : "Random Forest (V1)";

        const acuraciaTreino = dados.acuracia_treino_pct !== undefined
            ? Number(dados.acuracia_treino_pct)
            : (versao === "v2" ? 69.40 : 57.07);
        const acuraciaReal = Number(dados.acuracia_vencedor_pct);
        const ganhoSorte = dados.ganho_vs_aleatorio_pct !== undefined
            ? Number(dados.ganho_vs_aleatorio_pct)
            : (acuraciaReal - 33.33);

        metricasAtuaisCache[versao] = {
            acuraciaTreino,
            acuraciaReal,
            ganhoSorte,
            erroGols: dados.erro_medio_gols,
            totalJogos: dados.total_jogos_avaliados
        };

        elAcuraciaTreino.textContent = `${acuraciaTreino.toFixed(2)}%`;
        elAcuracia.textContent = `${acuraciaReal.toFixed(2)}%`;
        elGanhoSorte.textContent = `+${ganhoSorte.toFixed(2)}%`;
        elErroGols.textContent = `${dados.erro_medio_gols} gols`;
        elTotalJogos.textContent = `${dados.total_jogos_avaliados} jogos`;

        // Atualiza o selo do Card de Predição lá embaixo com a Acurácia Real e de Treino do Modelo
        if (valorConfianca) {
            valorConfianca.textContent = `${acuraciaReal.toFixed(2)}% (Treino: ${acuraciaTreino.toFixed(2)}%)`;
        }

        elRodape.textContent =
            `Modelo ${nomeModeloCurto} avaliado em ${dados.total_jogos_avaliados} partidas reais (${dados.periodo_avaliado}) ` +
            `• Acurácia Treino: ${acuraciaTreino.toFixed(2)}% • Acurácia Real: ${acuraciaReal.toFixed(2)}% • Erro médio (MAE): ${dados.erro_medio_gols} gols`;
    } catch (erro) {
        console.error("Falha ao carregar métricas:", erro);
        elRodape.textContent =
            "Servidor da API offline. Inicie o FastAPI com: uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload";
    }
}

/**
 * 5. Busca na API (GET /api/times) a lista de todos os clubes e preenche os dropdowns.
 */
async function carregarListaDeTimes(versao) {
    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");

    try {
        const resposta = await fetch(`${URL_BASE_API}/api/times?versao=${versao}`);
        if (!resposta.ok) {
            throw new Error("Não foi possível carregar a lista de times.");
        }

        const dados = await resposta.json();
        listaDeTimesCache = dados.times || [];

        // Limpa opções atuais
        selectCasa.innerHTML = "";
        selectFora.innerHTML = "";

        listaDeTimesCache.forEach((nomeTime) => {
            const opcaoCasa = document.createElement("option");
            opcaoCasa.value = nomeTime;
            opcaoCasa.textContent = nomeTime;
            selectCasa.appendChild(opcaoCasa);

            const opcaoFora = document.createElement("option");
            opcaoFora.value = nomeTime;
            opcaoFora.textContent = nomeTime;
            selectFora.appendChild(opcaoFora);
        });

        // Seleciona inicialmente os times vindos da URL ou Palmeiras vs Flamengo RJ como padrão
        const parametros = new URLSearchParams(window.location.search);
        const casaUrl = parametros.get("casa");
        const foraUrl = parametros.get("fora");

        let timeCasaPadrao = (casaUrl && listaDeTimesCache.includes(casaUrl))
            ? casaUrl
            : (listaDeTimesCache.includes("Palmeiras") ? "Palmeiras" : listaDeTimesCache[0] || "");

        let timeForaPadrao = (foraUrl && listaDeTimesCache.includes(foraUrl) && foraUrl !== timeCasaPadrao)
            ? foraUrl
            : (listaDeTimesCache.includes("Flamengo RJ") ? "Flamengo RJ" : listaDeTimesCache[1] || "");

        selectCasa.value = timeCasaPadrao;
        selectFora.value = timeForaPadrao;

        sincronizarBloqueioDeTimesIguais();
        atualizarLinkPainelTecnico();

        // Executa o primeiro cálculo automaticamente para preencher a tela inicial
        if (timeCasaPadrao && timeForaPadrao) {
            await calcularPalpiteComIA();
        }
    } catch (erro) {
        console.error("Erro ao carregar times:", erro);
        exibirAlerta("Não foi possível conectar à API em http://127.0.0.1:8000. Verifique se o servidor FastAPI está ligado.");
    }
}

/**
 * 6. Chamada quando o usuário troca qualquer time no dropdown.
 *    Impede que o mesmo clube seja escolhido nos dois lados.
 */
function aoTrocarTimeSelecionado() {
    sincronizarBloqueioDeTimesIguais();
    atualizarLinkPainelTecnico();
    ocultarAlerta();

    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");
    document.getElementById("legenda-mando-campo").textContent =
        `Confronto selecionado: ${selectCasa.value} (Mandante) vs ${selectFora.value} (Visitante)`;
}

function sincronizarBloqueioDeTimesIguais() {
    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");

    const valorCasa = selectCasa.value;
    const valorFora = selectFora.value;

    Array.from(selectCasa.options).forEach((opt) => {
        opt.disabled = opt.value === valorFora;
    });

    Array.from(selectFora.options).forEach((opt) => {
        opt.disabled = opt.value === valorCasa;
    });
}

/**
 * 7. Inverte instantaneamente o Time Mandante com o Time Visitante (Botão ⇄).
 */
async function inverterTimesSelecionados() {
    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");

    const tempCasa = selectCasa.value;
    selectCasa.value = selectFora.value;
    selectFora.value = tempCasa;

    sincronizarBloqueioDeTimesIguais();
    atualizarLinkPainelTecnico();
    await calcularPalpiteComIA();
}

/**
 * 8. Envia os dois clubes e a versão ativa para a API (POST /api/previsao).
 */
async function calcularPalpiteComIA() {
    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");
    const botaoCalcular = document.getElementById("btn-calcular-palpite");

    const timeCasa = selectCasa.value;
    const timeFora = selectFora.value;

    if (!timeCasa || !timeFora) {
        exibirAlerta("Selecione o time mandante e o time visitante.");
        return;
    }

    if (timeCasa === timeFora) {
        exibirAlerta("Escolha dois clubes diferentes para realizar a simulação.");
        return;
    }

    ocultarAlerta();
    botaoCalcular.disabled = true;
    botaoCalcular.textContent = "CALCULANDO PROBABILIDADES COM IA...";

    try {
        const resposta = await fetch(`${URL_BASE_API}/api/previsao`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                time_casa: timeCasa,
                time_fora: timeFora,
                versao: versaoAtual
            })
        });

        if (!resposta.ok) {
            throw new Error(`Erro HTTP ${resposta.status}`);
        }

        const dados = await resposta.json();
        atualizarInterfaceComPrevisao(timeCasa, timeFora, dados);
    } catch (erro) {
        console.error("Erro na predição:", erro);
        exibirAlerta("Erro ao calcular previsão. Verifique se o servidor Python (FastAPI) está rodando na porta 8000.");
    } finally {
        botaoCalcular.disabled = false;
        botaoCalcular.textContent = "CALCULAR PALPITE COM IA";
    }
}

/**
 * 9. Preenche todos os elementos visuais do Card de Predição e Raio-X com o retorno da API.
 */
function atualizarInterfaceComPrevisao(timeCasa, timeFora, dados) {
    const previsao = dados.previsao;
    const raioX = dados.raio_x;
    const retrospecto = dados.retrospecto;

    const nomeAlgoritmo = versaoAtual === "v2" ? "XGBoost V2" : "Random Forest V1";
    const met = metricasAtuaisCache[versaoAtual] || metricasAtuaisCache.v1;

    // Linha de resumo abaixo dos cards de seleção
    document.getElementById("legenda-mando-campo").textContent =
        `Mando de Campo: ${timeCasa} • Simulação processada via ${nomeAlgoritmo}`;

    // Cabeçalho da Predição
    document.getElementById("texto-palpite-ia").textContent =
        `Palpite (${nomeAlgoritmo}): ${previsao.palpite}`;

    // Selo de Acurácia do Modelo no Card de Predição (sempre exibindo a acurácia real + treino da versão ativa)
    const badgeConfianca = document.getElementById("badge-confianca-ia");
    const valorConfianca = document.getElementById("valor-confianca-ia");
    badgeConfianca.className = "badge-confianca confianca-alta";
    valorConfianca.textContent = `${met.acuraciaReal.toFixed(2)}% (Treino: ${met.acuraciaTreino.toFixed(2)}%)`;

    // Barras Horizontais de Probabilidade (Casa = Verde, Empate = Âmbar, Fora = Vermelho)
    document.getElementById("rotulo-barra-casa").textContent = `Vitória do ${timeCasa} (Mandante)`;
    document.getElementById("pct-barra-casa").textContent = `${previsao.chance_casa_pct.toFixed(1)}%`;
    document.getElementById("barra-progresso-casa").style.width = `${previsao.chance_casa_pct}%`;

    document.getElementById("pct-barra-empate").textContent = `${previsao.chance_empate_pct.toFixed(1)}%`;
    document.getElementById("barra-progresso-empate").style.width = `${previsao.chance_empate_pct}%`;

    document.getElementById("rotulo-barra-fora").textContent = `Vitória do ${timeFora} (Visitante)`;
    document.getElementById("pct-barra-fora").textContent = `${previsao.chance_fora_pct.toFixed(1)}%`;
    document.getElementById("barra-progresso-fora").style.width = `${previsao.chance_fora_pct}%`;

    // Cards Inferiores (Média de Gols Casa, Placar Mais Provável e Média de Gols Fora)
    const mediaHistoricaCasa = previsao.media_gols_casa !== undefined
        ? previsao.media_gols_casa
        : raioX.casa.gols_feitos_media;
    const mediaHistoricaFora = previsao.media_gols_fora !== undefined
        ? previsao.media_gols_fora
        : raioX.fora.gols_feitos_media;

    document.getElementById("titulo-media-casa").textContent = `Média de Gols (${timeCasa})`;
    document.getElementById("valor-media-casa").textContent = Number(mediaHistoricaCasa).toFixed(2);
    document.getElementById("detalhe-raiox-casa").textContent =
        `Últ. 5: ${raioX.casa.gols_feitos_media} pró • ${raioX.casa.gols_sofridos_media} sofr. • ${raioX.casa.pontos_media} pts`;

    // Placar Mais Provável formatado com "×" igual ao mockup_taste.jpg
    const placarFormatado = (previsao.placar_provavel || "1 x 1").replace("x", "×");
    document.getElementById("valor-placar-provavel").textContent = placarFormatado;
    document.getElementById("detalhe-total-gols").textContent =
        `Expectativa total: ${Number(previsao.total_gols_esperados).toFixed(2)} gols`;

    document.getElementById("titulo-media-fora").textContent = `Média de Gols (${timeFora})`;
    document.getElementById("valor-media-fora").textContent = Number(mediaHistoricaFora).toFixed(2);
    document.getElementById("detalhe-raiox-fora").textContent =
        `Últ. 5: ${raioX.fora.gols_feitos_media} pró • ${raioX.fora.gols_sofridos_media} sofr. • ${raioX.fora.pontos_media} pts`;

    // Faixa de Confronto Direto (H2H)
    document.getElementById("h2h-casa-pct").textContent = `${retrospecto.vitorias_casa_pct}%`;
    document.getElementById("h2h-empate-pct").textContent = `${retrospecto.empates_pct}%`;
    document.getElementById("h2h-fora-pct").textContent = `${retrospecto.vitorias_fora_pct}%`;
}

/**
 * 10. Funções auxiliares de alerta na tela.
 */
function exibirAlerta(mensagem) {
    const elAlerta = document.getElementById("mensagem-alerta");
    elAlerta.textContent = mensagem;
    elAlerta.classList.remove("oculto");
}

function ocultarAlerta() {
    const elAlerta = document.getElementById("mensagem-alerta");
    elAlerta.textContent = "";
    elAlerta.classList.add("oculto");
}

/**
 * 11. Sincroniza a URL do botão "Painel Técnico" com os times e versão atuais.
 */
function atualizarLinkPainelTecnico() {
    const linkPainel = document.getElementById("link-painel-tecnico");
    if (!linkPainel) return;
    const selectCasa = document.getElementById("select-time-casa");
    const selectFora = document.getElementById("select-time-fora");
    const timeCasa = selectCasa ? encodeURIComponent(selectCasa.value) : "";
    const timeFora = selectFora ? encodeURIComponent(selectFora.value) : "";
    linkPainel.href = `painel_tecnico.html?casa=${timeCasa}&fora=${timeFora}&versao=${versaoAtual}`;
}
