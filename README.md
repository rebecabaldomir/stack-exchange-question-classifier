#  Stack Exchange Question Classifier


Rebeca Andrade Baldomir  
Junho, 2018

### Proposta
Esse problema Stack Exchange Classifier é semelhante a vários desafios comuns de classificação de texto (https://www.hackerrank.com/challenges/stack-exchange-question-classifier/problem). Dado o texto da pergunta, você precisa identificar se ele foi retirado da seção eletrônica, android, scifi, gis, fotografia, etc. 

Solução baseada em https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

### Instalação

Este projeto requer o Python 2.7 e as seguintes bibliotecas do Python instaladas:

-   NumPy
-   json
-   matplotlib
-   scikit-learn

Também é necessário ter o software instalado para executar um Jupyter Notebook.

### Execução

Em uma janela de terminal ou linha de comando, navegue até o diretório de projeto que contém este README e execute o seguinte comando:

```
jupyter notebook stack-exchange-question-classifier.ipynb

```

Isso abrirá o software do Notebook Jupyter e o arquivo de projeto em seu navegador .

### Conjunto de Dados

A primeira linha será um inteiro N. N linhas seguem cada linha sendo um objeto JSON válido. Os seguintes campos de dados brutos são dados em json

- question (string): o texto no título da pergunta.
- excerpt (string): Trecho do corpo da pergunta.
- topic (string): o tópico sob o qual a pergunta foi postada.

