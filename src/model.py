from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAS, COLUNAS_MODELO, NUMERICAS


ROTULOS_VARIAVEIS = {
    "renda_mensal": "Renda mensal",
    "valor_solicitado": "Valor solicitado",
    "prazo_meses": "Prazo",
    "taxa_juros_mensal": "Taxa de juros",
    "parcela_estimada": "Parcela estimada",
    "comprometimento_renda": "Comprometimento da renda",
    "score_credito": "Score de crédito",
    "atrasos_12m": "Atrasos recentes",
    "tempo_emprego_meses": "Tempo no emprego",
    "reserva_meses": "Reserva financeira",
    "outros_creditos": "Outros créditos ativos",
    "relacao_credito_renda": "Crédito em relação à renda anual",
    "finalidade": "Finalidade do crédito",
    "moradia": "Tipo de moradia",
}


@dataclass
class ResultadoModelo:
    modelo: Pipeline
    metricas: dict[str, float]
    matriz_confusao: list[list[int]]
    y_teste: np.ndarray
    probabilidades_teste: np.ndarray


def criar_pipeline() -> Pipeline:
    preparo = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAS,
            ),
            ("numericas", StandardScaler(), NUMERICAS),
        ]
    )
    classificador = LogisticRegression(max_iter=2_000, random_state=42)
    return Pipeline([("preparo", preparo), ("modelo", classificador)])


def _calcular_metricas(
    y_real: np.ndarray | pd.Series,
    probabilidades: np.ndarray,
    limiar: float = 0.25,
) -> tuple[dict[str, float], list[list[int]]]:
    previsoes = (probabilidades >= limiar).astype(int)
    metricas = {
        "acuracia": accuracy_score(y_real, previsoes),
        "precisao": precision_score(y_real, previsoes, zero_division=0),
        "recall": recall_score(y_real, previsoes, zero_division=0),
        "f1": f1_score(y_real, previsoes, zero_division=0),
        "roc_auc": roc_auc_score(y_real, probabilidades),
        "brier": brier_score_loss(y_real, probabilidades),
    }
    matriz = confusion_matrix(y_real, previsoes, labels=[0, 1]).tolist()
    return metricas, matriz


def treinar_e_avaliar(dados: pd.DataFrame) -> ResultadoModelo:
    x = dados[COLUNAS_MODELO].copy()
    y = dados["inadimplente"].copy()
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    modelo_avaliacao = criar_pipeline()
    modelo_avaliacao.fit(x_treino, y_treino)
    probabilidades = modelo_avaliacao.predict_proba(x_teste)[:, 1]
    metricas, matriz = _calcular_metricas(y_teste, probabilidades)

    modelo_final = criar_pipeline()
    modelo_final.fit(x, y)
    return ResultadoModelo(
        modelo=modelo_final,
        metricas=metricas,
        matriz_confusao=matriz,
        y_teste=y_teste.to_numpy(),
        probabilidades_teste=probabilidades,
    )


def avaliar_limiar(
    resultado: ResultadoModelo,
    limiar: float,
) -> tuple[dict[str, float], list[list[int]]]:
    return _calcular_metricas(
        resultado.y_teste,
        resultado.probabilidades_teste,
        limiar,
    )


def prever_probabilidade(modelo: Pipeline, cenario: pd.DataFrame) -> float:
    return float(modelo.predict_proba(cenario)[0, 1])


def explicar_previsao(modelo: Pipeline, cenario: pd.DataFrame) -> pd.DataFrame:
    preparo = modelo.named_steps["preparo"]
    classificador = modelo.named_steps["modelo"]
    transformado = preparo.transform(cenario)
    if hasattr(transformado, "toarray"):
        transformado = transformado.toarray()

    nomes = preparo.get_feature_names_out()
    contribuicoes = transformado[0] * classificador.coef_[0]
    agrupadas: dict[str, float] = {}

    for nome, contribuicao in zip(nomes, contribuicoes, strict=True):
        nome_limpo = nome.split("__", maxsplit=1)[1]
        if nome.startswith("categoricas__"):
            variavel = next(
                coluna for coluna in CATEGORICAS if nome_limpo.startswith(f"{coluna}_")
            )
        else:
            variavel = nome_limpo
        agrupadas[variavel] = agrupadas.get(variavel, 0.0) + float(contribuicao)

    explicacao = pd.DataFrame(
        [
            {
                "variavel": ROTULOS_VARIAVEIS.get(variavel, variavel),
                "contribuicao": contribuicao,
                "efeito": "Aumenta o risco" if contribuicao > 0 else "Reduz o risco",
            }
            for variavel, contribuicao in agrupadas.items()
        ]
    )
    explicacao["magnitude"] = explicacao["contribuicao"].abs()
    return explicacao.sort_values("magnitude", ascending=False).head(8)
