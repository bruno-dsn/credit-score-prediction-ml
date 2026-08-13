from pathlib import Path

import numpy as np
import pandas as pd


COLUNAS_MODELO = [
    "renda_mensal",
    "valor_solicitado",
    "prazo_meses",
    "taxa_juros_mensal",
    "parcela_estimada",
    "comprometimento_renda",
    "score_credito",
    "atrasos_12m",
    "tempo_emprego_meses",
    "reserva_meses",
    "outros_creditos",
    "relacao_credito_renda",
    "finalidade",
    "moradia",
]

CATEGORICAS = ["finalidade", "moradia"]
NUMERICAS = [coluna for coluna in COLUNAS_MODELO if coluna not in CATEGORICAS]

FINALIDADES = [
    "Crédito pessoal",
    "Consolidação de dívidas",
    "Educação",
    "Financiamento de veículo",
    "Reforma ou serviços",
]

MORADIAS = ["Alugada", "Financiada", "Familiar", "Própria"]


def calcular_parcela_price(valor: float, taxa_mensal: float, prazo_meses: int) -> float:
    """Calcula a prestação fixa pelo Sistema Price."""
    taxa = taxa_mensal / 100
    if taxa == 0:
        return valor / prazo_meses
    fator = (1 + taxa) ** prazo_meses
    return valor * taxa * fator / (fator - 1)


def _sigmoide(valor: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-valor))


def gerar_dados_sinteticos(
    quantidade: int = 5_000,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera cenários fictícios, reproduzíveis e sem dados pessoais."""
    rng = np.random.default_rng(semente)

    renda = np.clip(
        rng.lognormal(mean=np.log(5_200), sigma=0.62, size=quantidade),
        1_500,
        35_000,
    )
    score = np.clip(300 + rng.beta(3.1, 2.4, quantidade) * 700, 300, 1_000)

    lambda_atrasos = np.clip(0.15 + (680 - score) / 190, 0.05, 2.8)
    atrasos = np.clip(rng.poisson(lambda_atrasos), 0, 8)
    tempo_emprego = np.clip(rng.gamma(2.1, 25, quantidade), 0, 360)
    reserva = np.clip(
        rng.gamma(1.35, 1.9, quantidade) + (renda >= 8_000) * rng.uniform(0, 2, quantidade),
        0,
        18,
    )
    outros_creditos = np.clip(rng.poisson(1.0, quantidade), 0, 6)

    finalidade = rng.choice(
        FINALIDADES,
        size=quantidade,
        p=[0.31, 0.16, 0.10, 0.25, 0.18],
    )
    moradia = rng.choice(
        MORADIAS,
        size=quantidade,
        p=[0.31, 0.27, 0.12, 0.30],
    )
    prazo = rng.choice(
        [6, 12, 18, 24, 36, 48, 60],
        size=quantidade,
        p=[0.05, 0.17, 0.12, 0.24, 0.22, 0.13, 0.07],
    )

    valor_base = rng.lognormal(mean=np.log(14_000), sigma=0.72, size=quantidade)
    limite_por_renda = renda * rng.uniform(2.0, 10.0, quantidade)
    valor = np.clip(np.minimum(valor_base, limite_por_renda), 1_000, 100_000)

    adicional_finalidade = pd.Series(finalidade).map(
        {
            "Crédito pessoal": 0.90,
            "Consolidação de dívidas": 1.25,
            "Educação": 0.25,
            "Financiamento de veículo": -0.25,
            "Reforma ou serviços": 0.45,
        }
    ).to_numpy()
    taxa = np.clip(
        1.05
        + adicional_finalidade
        + (690 - score) / 260
        + atrasos * 0.12
        + rng.normal(0, 0.28, quantidade),
        0.55,
        7.50,
    )

    parcela = np.array(
        [
            calcular_parcela_price(v, t, int(p))
            for v, t, p in zip(valor, taxa, prazo, strict=True)
        ]
    )
    comprometimento = parcela / renda * 100
    relacao_credito_renda = valor / (renda * 12)

    efeito_finalidade = pd.Series(finalidade).map(
        {
            "Crédito pessoal": 0.16,
            "Consolidação de dívidas": 0.48,
            "Educação": -0.10,
            "Financiamento de veículo": -0.22,
            "Reforma ou serviços": 0.04,
        }
    ).to_numpy()
    efeito_moradia = pd.Series(moradia).map(
        {"Alugada": 0.16, "Financiada": 0.10, "Familiar": 0.02, "Própria": -0.20}
    ).to_numpy()

    logit = (
        -1.95
        + (comprometimento - 28) * 0.052
        + (650 - score) * 0.0062
        + atrasos * 0.50
        + outros_creditos * 0.12
        - np.minimum(tempo_emprego, 120) * 0.0042
        - reserva * 0.16
        + relacao_credito_renda * 0.55
        + (taxa - 2.0) * 0.12
        + efeito_finalidade
        + efeito_moradia
        + rng.normal(0, 0.42, quantidade)
    )
    probabilidade = _sigmoide(logit)
    inadimplente = rng.binomial(1, probabilidade)

    return pd.DataFrame(
        {
            "renda_mensal": renda.round(2),
            "valor_solicitado": valor.round(2),
            "prazo_meses": prazo.astype(int),
            "taxa_juros_mensal": taxa.round(3),
            "parcela_estimada": parcela.round(2),
            "comprometimento_renda": comprometimento.round(2),
            "score_credito": score.round().astype(int),
            "atrasos_12m": atrasos.astype(int),
            "tempo_emprego_meses": tempo_emprego.round().astype(int),
            "reserva_meses": reserva.round(1),
            "outros_creditos": outros_creditos.astype(int),
            "relacao_credito_renda": relacao_credito_renda.round(3),
            "finalidade": finalidade,
            "moradia": moradia,
            "inadimplente": inadimplente.astype(int),
        }
    )


def carregar_dados(caminho: str | Path) -> pd.DataFrame:
    dados = pd.read_csv(caminho)
    faltantes = set(COLUNAS_MODELO + ["inadimplente"]) - set(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(faltantes)}")
    return dados


def montar_cenario(
    renda_mensal: float,
    valor_solicitado: float,
    prazo_meses: int,
    taxa_juros_mensal: float,
    score_credito: int,
    atrasos_12m: int,
    tempo_emprego_meses: int,
    reserva_meses: float,
    outros_creditos: int,
    finalidade: str,
    moradia: str,
) -> pd.DataFrame:
    parcela = calcular_parcela_price(valor_solicitado, taxa_juros_mensal, prazo_meses)
    comprometimento = parcela / renda_mensal * 100
    relacao_credito_renda = valor_solicitado / (renda_mensal * 12)

    return pd.DataFrame(
        [
            {
                "renda_mensal": renda_mensal,
                "valor_solicitado": valor_solicitado,
                "prazo_meses": prazo_meses,
                "taxa_juros_mensal": taxa_juros_mensal,
                "parcela_estimada": parcela,
                "comprometimento_renda": comprometimento,
                "score_credito": score_credito,
                "atrasos_12m": atrasos_12m,
                "tempo_emprego_meses": tempo_emprego_meses,
                "reserva_meses": reserva_meses,
                "outros_creditos": outros_creditos,
                "relacao_credito_renda": relacao_credito_renda,
                "finalidade": finalidade,
                "moradia": moradia,
            }
        ],
        columns=COLUNAS_MODELO,
    )
