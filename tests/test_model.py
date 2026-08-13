from src.data import gerar_dados_sinteticos, montar_cenario
from src.model import explicar_previsao, prever_probabilidade, treinar_e_avaliar


def test_modelo_treina_e_entrega_resultados_validos():
    dados = gerar_dados_sinteticos(quantidade=1_500, semente=42)
    resultado = treinar_e_avaliar(dados)

    assert 0.70 <= resultado.metricas["roc_auc"] <= 1
    assert 0 <= resultado.metricas["brier"] <= 1
    assert len(resultado.matriz_confusao) == 2


def test_previsao_e_explicacao_local():
    dados = gerar_dados_sinteticos(quantidade=1_200, semente=21)
    resultado = treinar_e_avaliar(dados)
    cenario = montar_cenario(
        renda_mensal=6_500,
        valor_solicitado=18_000,
        prazo_meses=24,
        taxa_juros_mensal=2.2,
        score_credito=720,
        atrasos_12m=0,
        tempo_emprego_meses=48,
        reserva_meses=4,
        outros_creditos=1,
        finalidade="Reforma ou serviços",
        moradia="Financiada",
    )

    probabilidade = prever_probabilidade(resultado.modelo, cenario)
    explicacao = explicar_previsao(resultado.modelo, cenario)

    assert 0 <= probabilidade <= 1
    assert not explicacao.empty
    assert {"variavel", "contribuicao", "efeito"}.issubset(explicacao.columns)
