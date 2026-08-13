from pathlib import Path

from src.data import gerar_dados_sinteticos


RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "clientes_credito_sinteticos.csv"


def main() -> None:
    dados = gerar_dados_sinteticos()
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(DESTINO, index=False)
    print(f"Arquivo criado: {DESTINO}")
    print(f"Registros: {len(dados)}")
    print(f"Inadimplência sintética: {dados['inadimplente'].mean():.1%}")


if __name__ == "__main__":
    main()
