
# Stack Exchange Question Classifier

Stack Exchange é um repositório de informações que possui 105 tópicos diferentes e cada tópico tem uma acervo de perguntas que foram feitas e respondidas por membros experientes da comunidade do StackExchange. Os tópicos são tão diversos quanto viagens, culinária, programação, engenharia e fotografia. Foram escolhidos 10 categoria de tópicos (gis, security, photo, mathematica, unix, wordpress, scifi, electronics, android, apple) e dada uma pergunta e um trecho, sua tarefa é identificar quais dentre os 10 tópicos a que pertence.


```python
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
```

### 1. Leitura e formatação dos dados


```python
data = json.load(open('training.json'))
```

Para executar algoritmos de aprendizado de máquina, precisamos converter os arquivos de texto em vetores de features numéricas. Nós estaremos usando o modelo 'bag of words' para o nosso exemplo. Resumidamente, nós segmentamos cada arquivo de texto em palavras (dividido pelo espaço) e contamos o número de vezes que cada palavra ocorre em cada documento e finalmente atribuímos a cada palavra um ID inteiro. Cada palavra única no nosso dicionário irá corresponder a uma característica 


```python
df = CountVectorizer(stop_words='english', strip_accents='unicode')
for line in data:
    data_test = [line['excerpt']]
    
for line in data:
    y = [line['topic']]

X_train = df.fit_transform(data_test)
X_train.shape
```




    (1, 14)



Selecionando cada tipo de variável (features e target) do conjunto de dados. Apenas contar o número de palavras em cada documento tem um problema: ele dará mais peso a documentos mais longos do que documentos mais curtos. Para evitar isso, podemos usar a frequência (TF - Term Frequencies) e, também, podemos até reduzir o peso de palavras mais comuns (TF-IDF - Term Frequency times inverse document frequency).


```python
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_train_tfidf.shape
```




    (1, 14)



Existem vários algoritmos que podem ser usados para classificação de texto, vamos usar Naive Bayes do sklearn.

### 2. Treinamento


```python
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y)
```

    /usr/local/lib/python2.7/dist-packages/sklearn/naive_bayes.py:461: RuntimeWarning: divide by zero encountered in log
      self.class_log_prior_ = (np.log(self.class_count_) -





    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



### 3. Teste


```python
data_test = json.load(open('input/input00.txt'))
```


```python
for line in data_test:
    test = [line['excerpt']]

X_test = df.transform(test)
```


```python
y_pred = naive_bayes.predict(X_test)
```
