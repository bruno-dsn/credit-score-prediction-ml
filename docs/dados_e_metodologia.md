# Dados e metodologia

## Por que a base é sintética

O projeto simula um laboratório brasileiro de risco de crédito, mas não utiliza cadastros de pessoas reais. Bases individuais de crédito exigem origem verificável, finalidade legítima, controles de privacidade e governança. Para um portfólio público, fabricar uma origem ou adaptar silenciosamente uma base estrangeira seria inadequado.

A solução foi criar 5.000 cenários sintéticos, reproduzíveis pela semente `42`. Nenhum registro corresponde a uma pessoa, empresa ou operação real.

## Variáveis

| Variável | Significado |
| --- | --- |
| `renda_mensal` | Renda informada em reais |
| `valor_solicitado` | Valor do crédito em reais |
| `prazo_meses` | Prazo contratual simulado |
| `taxa_juros_mensal` | Taxa mensal do cenário |
| `parcela_estimada` | Prestação calculada pelo Sistema Price |
| `comprometimento_renda` | Parcela dividida pela renda mensal |
| `score_credito` | Escala sintética de 300 a 1.000 |
| `atrasos_12m` | Quantidade de atrasos recentes |
| `tempo_emprego_meses` | Tempo no emprego atual |
| `reserva_meses` | Reserva financeira expressa em meses de renda |
| `outros_creditos` | Quantidade de créditos ativos |
| `relacao_credito_renda` | Crédito solicitado dividido pela renda anual |
| `finalidade` | Motivo informado para o crédito |
| `moradia` | Situação de moradia do cenário |
| `inadimplente` | Alvo sintético: 0 para em dia e 1 para inadimplente |

Sexo, raça, região, estado civil, nacionalidade e outros atributos sensíveis não são gerados nem usados.

## Geração do alvo

A probabilidade sintética é construída com uma função logística. Maior comprometimento da renda, score menor, atrasos, outros créditos e relação elevada entre crédito e renda aumentam a probabilidade. Reserva financeira e maior estabilidade no emprego reduzem a probabilidade. Um ruído aleatório impede que o alvo seja uma regra determinística.

Esse mecanismo cria um conjunto útil para estudar classificação e explicabilidade, mas não representa a inadimplência observada no Brasil.

## Modelo

O pipeline aplica:

1. codificação one-hot nas variáveis categóricas;
2. padronização nas variáveis numéricas;
3. regressão logística;
4. separação estratificada de 75% para treino e 25% para teste;
5. avaliação por ROC AUC, acurácia, precisão, recall, F1 e Brier Score.

O limiar inicial de 25% foi escolhido para equilibrar precisão e recall na amostra sintética. A aplicação permite alterá-lo e observar a matriz de confusão.

## Limitações

- probabilidades sem calibração em uma carteira real;
- ausência de validação temporal e externa;
- relações definidas para fins didáticos;
- inexistência de análise causal;
- uso proibido para aprovar, negar, recomendar ou precificar crédito.
