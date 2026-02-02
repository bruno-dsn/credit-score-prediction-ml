"""
Credit Score Predictor
Sistema de previsão de score de crédito utilizando Machine Learning

Autor: [Seu Nome]
Data: Fevereiro 2026
"""

import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import os


class CreditScorePredictor:
    """Classe para treinar e realizar previsões de score de crédito"""
    
    def __init__(self):
        self.modelo = None
        self.codificador_profissao = LabelEncoder()
        self.codificador_mix_credito = LabelEncoder()
        self.codificador_comportamento = LabelEncoder()
        
    def carregar_dados(self, caminho_arquivo):
        """Carrega dados do arquivo CSV"""
        print(f"Carregando dados de {caminho_arquivo}...")
        dados = pd.read_csv(caminho_arquivo)
        print(f"✓ {len(dados)} registros carregados")
        return dados
    
    def preprocessar_dados(self, dados, treino=True):
        """Realiza pré-processamento dos dados"""
        print("Pré-processando dados...")
        
        # Remover coluna id_cliente se existir
        if 'id_cliente' in dados.columns:
            dados = dados.drop(columns=['id_cliente'])
        
        # Codificar variáveis categóricas
        if treino:
            dados['profissao'] = self.codificador_profissao.fit_transform(
                dados['profissao']
            )
            dados['mix_credito'] = self.codificador_mix_credito.fit_transform(
                dados['mix_credito']
            )
            dados['comportamento_pagamento'] = self.codificador_comportamento.fit_transform(
                dados['comportamento_pagamento']
            )
        else:
            dados['profissao'] = self.codificador_profissao.transform(
                dados['profissao']
            )
            dados['mix_credito'] = self.codificador_mix_credito.transform(
                dados['mix_credito']
            )
            dados['comportamento_pagamento'] = self.codificador_comportamento.transform(
                dados['comportamento_pagamento']
            )
        
        print("✓ Pré-processamento concluído")
        return dados
    
    def treinar_modelo(self, dados):
        """Treina o modelo de Machine Learning"""
        print("\n" + "="*50)
        print("TREINAMENTO DO MODELO")
        print("="*50)
        
        # Separar features e target
        y = dados['score_credito']
        x = dados.drop(columns=['score_credito'])
        
        # Dividir em treino e teste
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x, y, test_size=0.3, random_state=42
        )
        
        print(f"Dados de treino: {len(x_treino)} registros")
        print(f"Dados de teste: {len(x_teste)} registros")
        
        # Treinar Random Forest
        print("\nTreinando Random Forest...")
        modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_rf.fit(x_treino, y_treino)
        previsao_rf = modelo_rf.predict(x_teste)
        acuracia_rf = accuracy_score(y_teste, previsao_rf)
        
        # Treinar KNN
        print("Treinando KNN...")
        modelo_knn = KNeighborsClassifier(n_neighbors=5)
        modelo_knn.fit(x_treino, y_treino)
        previsao_knn = modelo_knn.predict(x_teste)
        acuracia_knn = accuracy_score(y_teste, previsao_knn)
        
        # Selecionar melhor modelo
        print("\n" + "="*50)
        print("RESULTADOS")
        print("="*50)
        print(f"Acurácia Random Forest: {acuracia_rf:.4f} ({acuracia_rf*100:.2f}%)")
        print(f"Acurácia KNN: {acuracia_knn:.4f} ({acuracia_knn*100:.2f}%)")
        
        if acuracia_rf > acuracia_knn:
            print("\n✓ Melhor modelo: Random Forest")
            self.modelo = modelo_rf
        else:
            print("\n✓ Melhor modelo: KNN")
            self.modelo = modelo_knn
        
        print("="*50 + "\n")
        
        return self.modelo
    
    def prever(self, dados):
        """Realiza previsões em novos dados"""
        if self.modelo is None:
            raise ValueError("Modelo não treinado. Execute treinar_modelo() primeiro.")
        
        print("Realizando previsões...")
        previsoes = self.modelo.predict(dados)
        print("✓ Previsões concluídas")
        
        return previsoes
    
    def salvar_modelo(self, caminho='../models/modelo_final.pkl'):
        """Salva o modelo treinado"""
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        dados_modelo = {
            'modelo': self.modelo,
            'codificador_profissao': self.codificador_profissao,
            'codificador_mix_credito': self.codificador_mix_credito,
            'codificador_comportamento': self.codificador_comportamento
        }
        
        with open(caminho, 'wb') as f:
            pickle.dump(dados_modelo, f)
        
        print(f"✓ Modelo salvo em {caminho}")
    
    def carregar_modelo(self, caminho='../models/modelo_final.pkl'):
        """Carrega um modelo previamente treinado"""
        with open(caminho, 'rb') as f:
            dados_modelo = pickle.load(f)
        
        self.modelo = dados_modelo['modelo']
        self.codificador_profissao = dados_modelo['codificador_profissao']
        self.codificador_mix_credito = dados_modelo['codificador_mix_credito']
        self.codificador_comportamento = dados_modelo['codificador_comportamento']
        
        print(f"✓ Modelo carregado de {caminho}")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Previsão de Score de Crédito'
    )
    parser.add_argument(
        '--treinar',
        type=str,
        help='Caminho do arquivo CSV para treinar o modelo'
    )
    parser.add_argument(
        '--prever',
        type=str,
        help='Caminho do arquivo CSV com novos clientes'
    )
    parser.add_argument(
        '--modelo',
        type=str,
        default='../models/modelo_final.pkl',
        help='Caminho para salvar/carregar o modelo'
    )
    
    args = parser.parse_args()
    
    predictor = CreditScorePredictor()
    
    # Modo treino
    if args.treinar:
        dados = predictor.carregar_dados(args.treinar)
        dados = predictor.preprocessar_dados(dados, treino=True)
        predictor.treinar_modelo(dados)
        predictor.salvar_modelo(args.modelo)
    
    # Modo previsão
    elif args.prever:
        if os.path.exists(args.modelo):
            predictor.carregar_modelo(args.modelo)
        else:
            print("ERRO: Modelo não encontrado. Treine o modelo primeiro.")
            return
        
        novos_dados = predictor.carregar_dados(args.prever)
        novos_dados = predictor.preprocessar_dados(novos_dados, treino=False)
        previsoes = predictor.prever(novos_dados)
        
        print("\n" + "="*50)
        print("PREVISÕES")
        print("="*50)
        for i, score in enumerate(previsoes, 1):
            print(f"Cliente {i}: {score}")
        print("="*50)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
