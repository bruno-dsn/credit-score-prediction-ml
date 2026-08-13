from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src.data import (
    FINALIDADES,
    MORADIAS,
    carregar_dados,
    gerar_dados_sinteticos,
    montar_cenario,
)
from src.model import (
    avaliar_limiar,
    explicar_previsao,
    prever_probabilidade,
    treinar_e_avaliar,
)


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_DADOS = BASE_DIR / "data" / "clientes_credito_sinteticos.csv"

PERFIS = {
    "Equilibrado": {
        "renda_mensal": 6_500.0,
        "valor_solicitado": 18_000.0,
        "prazo_meses": 24,
        "taxa_juros_mensal": 2.2,
        "score_credito": 720,
        "atrasos_12m": 0,
        "tempo_emprego_meses": 48,
        "reserva_meses": 4.0,
        "outros_creditos": 1,
        "finalidade": "Reforma ou serviços",
        "moradia": "Financiada",
    },
    "Conservador": {
        "renda_mensal": 12_000.0,
        "valor_solicitado": 15_000.0,
        "prazo_meses": 18,
        "taxa_juros_mensal": 1.4,
        "score_credito": 830,
        "atrasos_12m": 0,
        "tempo_emprego_meses": 96,
        "reserva_meses": 8.0,
        "outros_creditos": 0,
        "finalidade": "Educação",
        "moradia": "Própria",
    },
    "Pressionado": {
        "renda_mensal": 3_500.0,
        "valor_solicitado": 25_000.0,
        "prazo_meses": 36,
        "taxa_juros_mensal": 4.2,
        "score_credito": 520,
        "atrasos_12m": 3,
        "tempo_emprego_meses": 8,
        "reserva_meses": 0.5,
        "outros_creditos": 3,
        "finalidade": "Consolidação de dívidas",
        "moradia": "Alugada",
    },
}

CORES_RISCO = {
    "Baixo": "#22C55E",
    "Moderado": "#F2C94C",
    "Elevado": "#F97316",
    "Muito elevado": "#EF4444",
}


@st.cache_data
def obter_dados() -> pd.DataFrame:
    if ARQUIVO_DADOS.exists():
        return carregar_dados(ARQUIVO_DADOS)
    return gerar_dados_sinteticos()


@st.cache_resource
def obter_resultado():
    return treinar_e_avaliar(obter_dados())


def moeda(valor: float) -> str:
    texto = f"{valor:,.2f}"
    return f"R$ {texto.replace(',', 'X').replace('.', ',').replace('X', '.')}"


def percentual(valor: float) -> str:
    return f"{valor:.1f}%".replace(".", ",")


def faixa_risco(probabilidade: float) -> str:
    if probabilidade < 0.12:
        return "Baixo"
    if probabilidade < 0.25:
        return "Moderado"
    if probabilidade < 0.45:
        return "Elevado"
    return "Muito elevado"


def aplicar_perfil() -> None:
    nome = st.session_state["perfil_pronto"]
    if nome in PERFIS:
        for chave, valor in PERFIS[nome].items():
            st.session_state[chave] = valor


def inicializar_estado() -> None:
    st.session_state.setdefault("perfil_pronto", "Equilibrado")
    for chave, valor in PERFIS["Equilibrado"].items():
        st.session_state.setdefault(chave, valor)


def criar_medidor(probabilidade: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probabilidade * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Probabilidade estimada de inadimplência"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": CORES_RISCO[faixa_risco(probabilidade)]},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 12], "color": "#163B2B"},
                    {"range": [12, 25], "color": "#4B421C"},
                    {"range": [25, 45], "color": "#512B17"},
                    {"range": [45, 100], "color": "#4A1F24"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 3},
                    "thickness": 0.8,
                    "value": probabilidade * 100,
                },
            },
        )
    ).update_layout(height=315, margin={"l": 30, "r": 30, "t": 60, "b": 10})


st.set_page_config(
    page_title="Laboratório de Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3 {letter-spacing: -0.025em;}
        [data-testid="stMetricValue"] {font-weight: 760;}
        .hero {
            padding: 1.55rem 1.75rem;
            margin-bottom: 1rem;
            border: 1px solid #24334D;
            border-radius: 18px;
            background: linear-gradient(120deg, #111C31 0%, #152B42 58%, #164E45 100%);
        }
        .hero h1 {margin: 0 0 .45rem 0; font-size: 2.35rem; color: #F8FAFC;}
        .hero p {margin: 0; color: #C8D5E5; font-size: 1.04rem; max-width: 850px;}
        .tag {
            display: inline-block;
            margin-bottom: .65rem;
            padding: .22rem .58rem;
            border-radius: 999px;
            background: #163B2B;
            color: #86EFAC;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .reading {
            padding: 1rem 1.1rem;
            border-left: 4px solid #38BDF8;
            border-radius: 8px;
            background: #111C2E;
            color: #D9E5F3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

inicializar_estado()

with st.sidebar:
    st.title("Configurações")
    st.selectbox(
        "Comece por um cenário",
        ["Equilibrado", "Conservador", "Pressionado", "Personalizado"],
        key="perfil_pronto",
        on_change=aplicar_perfil,
        help="Os três primeiros cenários preenchem o simulador automaticamente.",
    )

    st.subheader("Pessoa solicitante")
    st.number_input(
        "Renda mensal",
        min_value=1_500.0,
        max_value=50_000.0,
        step=500.0,
        format="%.2f",
        key="renda_mensal",
    )
    st.slider("Score de crédito", 300, 1_000, key="score_credito")
    st.slider("Atrasos nos últimos 12 meses", 0, 8, key="atrasos_12m")
    st.slider("Tempo no emprego atual (meses)", 0, 240, key="tempo_emprego_meses")
    st.slider(
        "Reserva financeira (meses de renda)",
        0.0,
        18.0,
        step=0.5,
        key="reserva_meses",
    )
    st.slider("Outros créditos ativos", 0, 6, key="outros_creditos")
    st.selectbox("Tipo de moradia", MORADIAS, key="moradia")

    st.subheader("Crédito solicitado")
    st.number_input(
        "Valor solicitado",
        min_value=1_000.0,
        max_value=100_000.0,
        step=1_000.0,
        format="%.2f",
        key="valor_solicitado",
    )
    st.select_slider(
        "Prazo",
        options=[6, 12, 18, 24, 36, 48, 60],
        format_func=lambda valor: f"{valor} meses",
        key="prazo_meses",
    )
    st.slider(
        "Taxa de juros ao mês",
        0.5,
        7.5,
        step=0.1,
        format="%.1f%%",
        key="taxa_juros_mensal",
    )
    st.selectbox("Finalidade", FINALIDADES, key="finalidade")
    st.caption(
        "Simulação educacional com dados sintéticos. Nenhum dado pessoal real é utilizado."
    )

cenario = montar_cenario(
    renda_mensal=st.session_state["renda_mensal"],
    valor_solicitado=st.session_state["valor_solicitado"],
    prazo_meses=st.session_state["prazo_meses"],
    taxa_juros_mensal=st.session_state["taxa_juros_mensal"],
    score_credito=st.session_state["score_credito"],
    atrasos_12m=st.session_state["atrasos_12m"],
    tempo_emprego_meses=st.session_state["tempo_emprego_meses"],
    reserva_meses=st.session_state["reserva_meses"],
    outros_creditos=st.session_state["outros_creditos"],
    finalidade=st.session_state["finalidade"],
    moradia=st.session_state["moradia"],
)

dados = obter_dados()
resultado = obter_resultado()
probabilidade = prever_probabilidade(resultado.modelo, cenario)
faixa = faixa_risco(probabilidade)
parcela = float(cenario.loc[0, "parcela_estimada"])
comprometimento = float(cenario.loc[0, "comprometimento_renda"])

st.markdown(
    """
    <section class="hero">
        <div class="tag">Machine Learning aplicado a crédito</div>
        <h1>Laboratório de Risco de Crédito</h1>
        <p>Monte um cenário em reais, acompanhe a probabilidade estimada de inadimplência e entenda quais fatores influenciaram o resultado.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Projeto educacional com dados sintéticos inspirados no contexto brasileiro. "
    "Não representa política de concessão, recomendação financeira ou decisão de crédito."
)

aba_geral, aba_sensibilidade, aba_modelo, aba_dados = st.tabs(
    [
        "Visão geral",
        "Análise de sensibilidade",
        "Modelo e métricas",
        "Dados e metodologia",
    ]
)

with aba_geral:
    coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
    coluna_1.metric("Probabilidade de inadimplência", percentual(probabilidade * 100))
    coluna_2.metric("Faixa de risco", faixa)
    coluna_3.metric("Parcela estimada", moeda(parcela))
    coluna_4.metric("Renda comprometida", percentual(comprometimento))

    esquerda, direita = st.columns([1, 1.1], gap="large")
    with esquerda:
        st.plotly_chart(criar_medidor(probabilidade), use_container_width=True)
        aproximacao = round(probabilidade * 100)
        st.markdown(
            f"""
            <div class="reading">
                Em uma leitura simplificada, o modelo estima que cerca de <strong>{aproximacao} em cada 100</strong> perfis sintéticos semelhantes apresentariam inadimplência. Essa é uma estimativa estatística, não uma previsão individual garantida.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with direita:
        st.subheader("Fatores que influenciaram este cenário")
        explicacao = explicar_previsao(resultado.modelo, cenario).sort_values(
            "contribuicao"
        )
        grafico_explicacao = px.bar(
            explicacao,
            x="contribuicao",
            y="variavel",
            orientation="h",
            color="efeito",
            color_discrete_map={
                "Aumenta o risco": "#EF4444",
                "Reduz o risco": "#22C55E",
            },
            labels={
                "contribuicao": "Impacto relativo na estimativa",
                "variavel": "",
                "efeito": "Efeito",
            },
        )
        grafico_explicacao.add_vline(x=0, line_color="#94A3B8", line_width=1)
        grafico_explicacao.update_layout(height=420, legend_title_text="")
        st.plotly_chart(grafico_explicacao, use_container_width=True)
        st.caption(
            "Barras vermelhas aumentam a estimativa de risco; barras verdes reduzem. "
            "O tamanho mostra a força relativa dentro do modelo."
        )

    st.subheader("Resumo do cenário")
    resumo = pd.DataFrame(
        {
            "Informação": [
                "Renda mensal",
                "Valor solicitado",
                "Prazo",
                "Taxa mensal",
                "Parcela estimada",
                "Comprometimento da renda",
                "Score",
                "Atrasos recentes",
                "Reserva financeira",
                "Finalidade",
            ],
            "Valor": [
                moeda(cenario.loc[0, "renda_mensal"]),
                moeda(cenario.loc[0, "valor_solicitado"]),
                f"{int(cenario.loc[0, 'prazo_meses'])} meses",
                percentual(cenario.loc[0, "taxa_juros_mensal"]),
                moeda(cenario.loc[0, "parcela_estimada"]),
                percentual(cenario.loc[0, "comprometimento_renda"]),
                str(int(cenario.loc[0, "score_credito"])),
                str(int(cenario.loc[0, "atrasos_12m"])),
                f"{cenario.loc[0, 'reserva_meses']:.1f} meses".replace(".", ","),
                cenario.loc[0, "finalidade"],
            ],
        }
    )
    st.dataframe(resumo, hide_index=True, use_container_width=True)
    st.download_button(
        "Baixar cenário em CSV",
        data=cenario.to_csv(index=False).encode("utf-8"),
        file_name="cenario_risco_credito.csv",
        mime="text/csv",
    )

with aba_sensibilidade:
    st.header("Como o resultado muda?")
    st.write(
        "A análise de sensibilidade altera uma variável por vez e mantém as demais constantes. "
        "Isso ajuda a entender o comportamento do modelo sem tratar correlação como causa."
    )

    valores = [
        max(1_000.0, st.session_state["valor_solicitado"] * fator)
        for fator in [0.50, 0.65, 0.80, 1.00, 1.20, 1.35, 1.50]
    ]
    linhas_valor = []
    for valor in valores:
        alternativa = montar_cenario(
            renda_mensal=st.session_state["renda_mensal"],
            valor_solicitado=valor,
            prazo_meses=st.session_state["prazo_meses"],
            taxa_juros_mensal=st.session_state["taxa_juros_mensal"],
            score_credito=st.session_state["score_credito"],
            atrasos_12m=st.session_state["atrasos_12m"],
            tempo_emprego_meses=st.session_state["tempo_emprego_meses"],
            reserva_meses=st.session_state["reserva_meses"],
            outros_creditos=st.session_state["outros_creditos"],
            finalidade=st.session_state["finalidade"],
            moradia=st.session_state["moradia"],
        )
        linhas_valor.append(
            {
                "Valor solicitado": valor,
                "Probabilidade": prever_probabilidade(resultado.modelo, alternativa) * 100,
            }
        )

    scores = list(range(350, 951, 50))
    linhas_score = []
    for score in scores:
        alternativa = cenario.copy()
        alternativa.loc[0, "score_credito"] = score
        linhas_score.append(
            {
                "Score": score,
                "Probabilidade": prever_probabilidade(resultado.modelo, alternativa) * 100,
            }
        )

    sensibilidade_1, sensibilidade_2 = st.columns(2, gap="large")
    with sensibilidade_1:
        grafico_valor = px.line(
            pd.DataFrame(linhas_valor),
            x="Valor solicitado",
            y="Probabilidade",
            markers=True,
            labels={"Probabilidade": "Probabilidade estimada (%)"},
            title="Efeito do valor solicitado",
        )
        grafico_valor.update_traces(line_color="#38BDF8")
        grafico_valor.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
        grafico_valor.update_yaxes(ticksuffix="%", rangemode="tozero")
        st.plotly_chart(grafico_valor, use_container_width=True)

    with sensibilidade_2:
        grafico_score = px.line(
            pd.DataFrame(linhas_score),
            x="Score",
            y="Probabilidade",
            markers=True,
            labels={"Probabilidade": "Probabilidade estimada (%)"},
            title="Efeito do score de crédito",
        )
        grafico_score.update_traces(line_color="#22C55E")
        grafico_score.update_yaxes(ticksuffix="%", rangemode="tozero")
        st.plotly_chart(grafico_score, use_container_width=True)

    st.warning(
        "Os gráficos mostram associações aprendidas pelo modelo sintético. "
        "Eles não indicam que alterar isoladamente uma variável produziria o mesmo efeito no mundo real."
    )

with aba_modelo:
    st.header("Modelo e métricas")
    st.write(
        "A regressão logística foi treinada em 75% da base e avaliada nos 25% restantes. "
        "O controle abaixo mostra como o limiar de classificação altera os erros do modelo."
    )
    limiar_percentual = st.slider(
        "Limiar para classificar inadimplência",
        min_value=10,
        max_value=70,
        value=25,
        step=5,
        format="%d%%",
    )
    limiar = limiar_percentual / 100
    metricas_limiar, matriz_limiar = avaliar_limiar(resultado, limiar)

    metrica_1, metrica_2, metrica_3, metrica_4 = st.columns(4)
    metrica_1.metric("ROC AUC", f"{resultado.metricas['roc_auc']:.3f}")
    metrica_2.metric("Acurácia", f"{metricas_limiar['acuracia']:.1%}")
    metrica_3.metric("Precisão", f"{metricas_limiar['precisao']:.1%}")
    metrica_4.metric("Recall", f"{metricas_limiar['recall']:.1%}")

    grafico_1, grafico_2 = st.columns(2, gap="large")
    with grafico_1:
        matriz = go.Figure(
            go.Heatmap(
                z=matriz_limiar,
                x=["Previsto: em dia", "Previsto: inadimplente"],
                y=["Real: em dia", "Real: inadimplente"],
                colorscale=[[0, "#172033"], [1, "#38BDF8"]],
                showscale=False,
                text=matriz_limiar,
                texttemplate="%{text}",
                textfont={"size": 20},
            )
        )
        matriz.update_layout(
            title="Matriz de confusão",
            height=420,
            margin={"l": 20, "r": 20, "t": 65, "b": 30},
        )
        st.plotly_chart(matriz, use_container_width=True)

    with grafico_2:
        falso_positivo, verdadeiro_positivo, _ = roc_curve(
            resultado.y_teste,
            resultado.probabilidades_teste,
        )
        curva = go.Figure()
        curva.add_trace(
            go.Scatter(
                x=falso_positivo,
                y=verdadeiro_positivo,
                mode="lines",
                name=f"Modelo (AUC {resultado.metricas['roc_auc']:.3f})",
                line={"color": "#22C55E", "width": 3},
            )
        )
        curva.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Referência aleatória",
                line={"color": "#94A3B8", "dash": "dash"},
            )
        )
        curva.update_layout(
            title="Curva ROC",
            xaxis_title="Taxa de falsos positivos",
            yaxis_title="Taxa de verdadeiros positivos",
            height=420,
        )
        st.plotly_chart(curva, use_container_width=True)

    with st.expander("Como interpretar as métricas"):
        st.markdown(
            """
            - **ROC AUC:** capacidade de ordenar perfis de menor e maior risco em vários limiares.
            - **Acurácia:** proporção total de classificações corretas.
            - **Precisão:** entre os casos sinalizados, quantos eram inadimplentes.
            - **Recall:** entre os inadimplentes, quantos foram identificados.
            - **Brier Score:** mede a qualidade das probabilidades; valores menores são melhores.
            """
        )
        st.write(f"Brier Score no conjunto de teste: **{resultado.metricas['brier']:.3f}**")

with aba_dados:
    st.header("Dados e metodologia")
    taxa_base = dados["inadimplente"].mean()
    total_1, total_2, total_3 = st.columns(3)
    total_1.metric("Cenários sintéticos", f"{len(dados):,}".replace(",", "."))
    total_2.metric("Variáveis do modelo", "14")
    total_3.metric("Inadimplência na amostra", f"{taxa_base:.1%}")

    dados_exibicao = dados.assign(
        situacao=dados["inadimplente"].map({0: "Em dia", 1: "Inadimplente"})
    )
    grafico_dados_1, grafico_dados_2 = st.columns(2, gap="large")
    with grafico_dados_1:
        distribuicao = px.histogram(
            dados_exibicao,
            x="score_credito",
            color="situacao",
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={"Em dia": "#22C55E", "Inadimplente": "#EF4444"},
            labels={"score_credito": "Score de crédito", "situacao": "Situação"},
            title="Distribuição do score por situação",
        )
        st.plotly_chart(distribuicao, use_container_width=True)

    with grafico_dados_2:
        faixas_comprometimento = pd.cut(
            dados_exibicao["comprometimento_renda"],
            bins=[0, 10, 20, 30, 40, 60, float("inf")],
            labels=["Até 10%", "10–20%", "20–30%", "30–40%", "40–60%", "Acima de 60%"],
            include_lowest=True,
        )
        por_faixa = (
            dados_exibicao.assign(faixa=faixas_comprometimento)
            .groupby("faixa", observed=True)["inadimplente"]
            .mean()
            .mul(100)
            .reset_index(name="Taxa de inadimplência")
        )
        comprometimento_fig = px.bar(
            por_faixa,
            x="faixa",
            y="Taxa de inadimplência",
            text_auto=".1f",
            labels={"faixa": "Comprometimento da renda"},
            title="Inadimplência sintética por comprometimento",
        )
        comprometimento_fig.update_traces(marker_color="#38BDF8")
        comprometimento_fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(comprometimento_fig, use_container_width=True)

    st.subheader("O que torna a base brasileira?")
    st.write(
        "Os cenários usam valores em reais e variáveis familiares ao crédito no Brasil: renda mensal, "
        "score de 300 a 1.000, atrasos recentes, comprometimento da renda, reserva financeira, "
        "taxa mensal, prazo e finalidade do crédito. Nenhum registro representa uma pessoa real."
    )

    st.subheader("Limites e cuidados")
    st.markdown(
        """
        - A base é sintética e serve para demonstrar o fluxo de Ciência de Dados.
        - As probabilidades não foram calibradas com uma carteira de crédito real.
        - Sexo, raça, região, estado civil e outros atributos sensíveis não foram usados.
        - Uma aplicação real exigiria dados representativos, validação temporal, monitoramento de deriva, governança e auditoria de equidade.
        - O simulador não aprova, reprova, precifica ou recomenda crédito.
        """
    )

st.divider()
st.caption(
    "Laboratório educacional de Machine Learning. Dados sintéticos, modelo reproduzível e explicações locais."
)
