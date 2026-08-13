from src.data import COLUNAS_MODELO, gerar_dados_sinteticos, montar_cenario


def test_base_sintetica_e_reproduzivel():
    primeira = gerar_dados_sinteticos(quantidade=300, semente=7)
    segunda = gerar_dados_sinteticos(quantidade=300, semente=7)

    assert primeira.equals(segunda)
    assert primeira.shape == (300, 15)
    assert set(COLUNAS_MODELO).issubset(primeira.columns)
    assert primeira["inadimplente"].isin([0, 1]).all()


def test_cenario_calcula_parcela_e_comprometimento():
    cenario = montar_cenario(
        renda_mensal=6_000,
        valor_solicitado=15_000,
        prazo_meses=24,
        taxa_juros_mensal=2.0,
        score_credito=700,
        atrasos_12m=0,
        tempo_emprego_meses=36,
        reserva_meses=3,
        outros_creditos=1,
        finalidade="Crédito pessoal",
        moradia="Alugada",
    )

    assert cenario.loc[0, "parcela_estimada"] > 0
    assert cenario.loc[0, "comprometimento_renda"] > 0
    assert list(cenario.columns) == COLUNAS_MODELO
