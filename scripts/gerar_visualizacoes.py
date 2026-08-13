from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.data import carregar_dados
from src.model import treinar_e_avaliar


RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    dados = carregar_dados(RAIZ / "data" / "clientes_credito_sinteticos.csv")
    resultado = treinar_e_avaliar(dados)

    plt.style.use("dark_background")
    figura, eixos = plt.subplots(1, 3, figsize=(16, 5.2))
    figura.patch.set_facecolor("#0B1120")

    metricas = ["ROC AUC", "Acurácia", "Precisão", "Recall"]
    valores = [
        resultado.metricas["roc_auc"],
        resultado.metricas["acuracia"],
        resultado.metricas["precisao"],
        resultado.metricas["recall"],
    ]
    cores = ["#38BDF8", "#22C55E", "#F2C94C", "#F97316"]
    eixos[0].barh(metricas[::-1], valores[::-1], color=cores[::-1])
    eixos[0].set_xlim(0, 1)
    eixos[0].set_title("Métricas no conjunto de teste", fontweight="bold")
    eixos[0].grid(axis="x", alpha=0.15)
    for indice, valor in enumerate(valores[::-1]):
        eixos[0].text(valor + 0.02, indice, f"{valor:.1%}", va="center")

    matriz = np.array(resultado.matriz_confusao)
    imagem = eixos[1].imshow(matriz, cmap="Blues")
    eixos[1].set_xticks([0, 1], ["Em dia", "Inadimplente"])
    eixos[1].set_yticks([0, 1], ["Em dia", "Inadimplente"])
    eixos[1].set_xlabel("Previsto")
    eixos[1].set_ylabel("Real")
    eixos[1].set_title("Matriz de confusão (limiar 25%)", fontweight="bold")
    for linha in range(2):
        for coluna in range(2):
            eixos[1].text(
                coluna,
                linha,
                str(matriz[linha, coluna]),
                ha="center",
                va="center",
                fontsize=15,
                color="white" if matriz[linha, coluna] > matriz.max() / 2 else "#0B1120",
            )
    figura.colorbar(imagem, ax=eixos[1], fraction=0.046, pad=0.04)

    falso_positivo, verdadeiro_positivo, _ = roc_curve(
        resultado.y_teste,
        resultado.probabilidades_teste,
    )
    eixos[2].plot(
        falso_positivo,
        verdadeiro_positivo,
        color="#22C55E",
        linewidth=2.5,
        label=f"Modelo (AUC {resultado.metricas['roc_auc']:.3f})",
    )
    eixos[2].plot([0, 1], [0, 1], "--", color="#94A3B8", label="Aleatório")
    eixos[2].set_xlabel("Taxa de falsos positivos")
    eixos[2].set_ylabel("Taxa de verdadeiros positivos")
    eixos[2].set_title("Curva ROC", fontweight="bold")
    eixos[2].legend(loc="lower right")
    eixos[2].grid(alpha=0.15)

    for eixo in eixos:
        eixo.set_facecolor("#111827")
        for borda in eixo.spines.values():
            borda.set_color("#334155")

    figura.suptitle(
        "Laboratório brasileiro de risco de crédito",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    figura.tight_layout()
    destino = RAIZ / "assets" / "avaliacao_modelo.png"
    figura.savefig(destino, dpi=180, bbox_inches="tight", facecolor=figura.get_facecolor())
    plt.close(figura)
    print(f"Imagem criada: {destino}")


if __name__ == "__main__":
    main()
