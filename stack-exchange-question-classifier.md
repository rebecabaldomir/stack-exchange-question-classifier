
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
data = pd.read_json('training.json', orient='columns')
print data
```

                                                     excerpt  \
    0      I'm trying to work out, in general terms, the ...   
    1      Can I know which component senses heat or acts...   
    2      I am replacing a wall outlet with a Cooper Wir...   
    3      i have been reading about the buck converter, ...   
    4      I need help with deciding on a Master's Projec...   
    5      Is it possible to supply power to a very high ...   
    6      My solenoid is part of an old espresso machine...   
    7      I use IP core PCI-E for Virtex-6 v.2.5\n\nTher...   
    8      I want to replace an ON Semicondutor BC557 PNP...   
    9      I will be using the AT32UC3C2512C, the AVR3276...   
    10     So I had an old gen 1 ipad that my son tried t...   
    11     I flashed the router with OpenWRT and it worke...   
    12     I posted a similar question a while ago and go...   
    13     I need to buy a voltage converter(adapter) for...   
    14     I had purchased an ATMEGA 16 microcontroller a...   
    15     My computer AVR's switch just got broken. Do y...   
    16     I'm looking into creating a solution to monito...   
    17     I am a new to electronics. I am trying to setu...   
    18     If I have an oscillator at 10 kHz outputting 5...   
    19     Using a packet sniffer I am able to detect and...   
    20     I'm sitting an exam on Computer Architecture i...   
    21     Our school decided to go and buy some good CAD...   
    22     I read that the frequencies over which FM sign...   
    23     I need RS232 IC with Features: power supply 3....   
    24     I have a product that was designed to sell in ...   
    25     My apologies in advance if this is not a good ...   
    26     I'm struggling a bit with the following calcul...   
    27     I am looking for surface mount LEDs that have ...   
    28     At higher temperatures, will computers get fas...   
    29     This question is based on another question sub...   
    ...                                                  ...   
    20189  I'm considering rebuilding a drupal site in wo...   
    20190  I have to add a variable value myvarvalue to e...   
    20191  I want to "save" my menus into transients. I w...   
    20192  How do I place the following javaScript in the...   
    20193  I'm running a wordpress blog. \nMy blog is ver...   
    20194  I have a custom post type where I have testimo...   
    20195  I'm trying to configure the comments screen in...   
    20196  I would like to know, what is the best practic...   
    20197  please i am wondering if there was a plugin or...   
    20198  Hi I need to make automated price updater of m...   
    20199  WordPress makes it easy to embed youtube video...   
    20200  I'm trying to create my own custom fields meta...   
    20201  I only want to show up projects with specific ...   
    20202  I installed bbPress and for the most part it w...   
    20203  Ok, so I'm totally confused on this and am loo...   
    20204  I'm trying to adjust the comments screen for n...   
    20205  I have a Custom Post Type of "projects", that ...   
    20206  I am new in PHP and WordPress and I am persona...   
    20207  The following code loops through posts, and th...   
    20208  I've got a small challenge,  right now I have ...   
    20209  I created a custom field called 'Page Descript...   
    20210  I’m doing up a photography theme in WP at the ...   
    20211  Trying to filter out specific posts that have ...   
    20212  I am using the below mentioned code to get top...   
    20213  So here is my problem, I modified a starkers t...   
    20214  I have a Custom Post Type called Recipe with p...   
    20215  I'm using the code below to track when a user ...   
    20216  add_action( 'pre_get_posts', 'custom_pre_get_p...   
    20217  i have wordpress blog with many posts. each po...   
    20218  I have many issues with the use of rewriting, ...   
    
                                                    question        topic  
    0      What is the effective differencial effective o...  electronics  
    1                           Heat sensor with fan cooling  electronics  
    2      Outlet Installation--more wires than my new ou...  electronics  
    3                      Buck Converter Operation Question  electronics  
    4      Urgent help in area of ASIC design, verificati...  electronics  
    5             Slowly supplying power to a very high load  electronics  
    6      I have a 110 VAC solenoid and want to know wha...  electronics  
    7      Can't read user-defined configuration space pc...  electronics  
    8      Understanding specs of PNP transistor for repl...  electronics  
    9                        MCU crystal capacitor selection  electronics  
    10         Manually recharging an ipad battery [on hold]  electronics  
    11     Wr703n Soldered Serial line, might have burned...  electronics  
    12                      Solar panel not charging battery  electronics  
    13     Device labeled 220/230V, 50Hz, 35V-A. What wat...  electronics  
    14               _delay_ms not working with ATmega 16/32  electronics  
    15     Is it alright if I'll just connect the black a...  electronics  
    16       Proper way to monitor temperature of a surface?  electronics  
    17                AttatchInterrupt Constantly interrupts  electronics  
    18                                     RF power envelope  electronics  
    19     how do you hook into a live stream of 802.15.4...  electronics  
    20       Functional Unit and Micro-operations Schematics  electronics  
    21     Collecting best CAD software for drawing elect...  electronics  
    22                    What decides the range of FM band?  electronics  
    23     Is possible use driver and receiver from diffe...  electronics  
    24     Can I sell a CSA approved product in the US at...  electronics  
    25     Is it possible to modify an HDMI/coax/PC video...  electronics  
    26                            Node transmission per hour  electronics  
    27     Surface Mount LEDs that are similar to these t...  electronics  
    28         Do computers speed up at higher temperatures?  electronics  
    29                      AMD/Intel CPU Yield/Failure Rate  electronics  
    ...                                                  ...          ...  
    20189  Can multiple custom post types share a custom ...    wordpress  
    20190              Add query var to every post permalink    wordpress  
    20191  Is there an action for save_menu and/or update...    wordpress  
    20192       javaScript in &lt;head&gt; section of WP API    wordpress  
    20193             HTTP Code 302 on Google Webmaster tool    wordpress  
    20194  WP_Query - How to show last post from each met...    wordpress  
    20195  Comments screen in backend, how to disable ema...    wordpress  
    20196        Best practice to update the file header.php    wordpress  
    20197  How to add thumbnails to posts and pages autom...    wordpress  
    20198  I need to create a dashboard widget with input...    wordpress  
    20199              Auto-centering youtube videos in post    wordpress  
    20200  Custom fields: my custom checkbox area doesn't...    wordpress  
    20201  Echo Custom Post Types - show project with tax...    wordpress  
    20202          Setup login/register buttons for bbPress?    wordpress  
    20203  Weird Code Being Added to Wordpress Site [Thesis]    wordpress  
    20204  Comments screen in backend, how to disable Qui...    wordpress  
    20205  How to List Parent Term Links for Custom Taxon...    wordpress  
    20206  Some problems calling a function into sprintf(...    wordpress  
    20207  Looping through textfield submission and savin...    wordpress  
    20208                Change Image Sizes for Mobile Theme    wordpress  
    20209        Custom fields won't display on my blog page    wordpress  
    20210  Hiding or removing file extension displayed in...    wordpress  
    20211           filter posts by meta key with pagination    wordpress  
    20212        Adding Multiple “Parents” in get_categories    wordpress  
    20213  Page content not indexed in the Wordpress sear...    wordpress  
    20214  How to set a Custom Post Type as the parent of...    wordpress  
    20215                 Tracking last login and last visit    wordpress  
    20216  How to exclude the particular category from th...    wordpress  
    20217  display sub categories assoccited with each po...    wordpress  
    20218       Lost of query parameter when using permalink    wordpress  
    
    [20219 rows x 3 columns]
    

Para executar algoritmos de aprendizado de máquina, precisamos converter os arquivos de texto em vetores de features numéricas. Nós estaremos usando o modelo 'bag of words' para o nosso exemplo. Resumidamente, nós segmentamos cada arquivo de texto em palavras (dividido pelo espaço) e contamos o número de vezes que cada palavra ocorre em cada documento e finalmente atribuímos a cada palavra um ID inteiro. Cada palavra única no nosso dicionário irá corresponder a uma característica 


```python
df = CountVectorizer(stop_words='english', strip_accents='unicode')

data_test = data['excerpt']
y = data['topic']

X_train = df.fit_transform(data_test)
X_train.shape
```




    (20219, 29135)



Selecionando cada tipo de variável (features e target) do conjunto de dados. Apenas contar o número de palavras em cada documento tem um problema: ele dará mais peso a documentos mais longos do que documentos mais curtos. Para evitar isso, podemos usar a frequência (TF - Term Frequencies) e, também, podemos até reduzir o peso de palavras mais comuns (TF-IDF - Term Frequency times inverse document frequency).


```python
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_train_tfidf.shape
```




    (20219, 29135)



Existem vários algoritmos que podem ser usados para classificação de texto, vamos usar Naive Bayes do sklearn.

### 2. Treinamento


```python
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



### 3. Teste


```python
#data_test_json = json.load(open('input00.txt'))
data_test_json = pd.read_json('input00.txt', orient='columns')
```


```python
X = data_test_json['excerpt']
#for line in data_test:
#    test = [line['excerpt']]

X_test = df.transform(X)
```


```python
y_pred = naive_bayes.predict(X_test)
```


```python

import pandas as pd

#output = json.load(open('output00.txt'))
y_true = pd.read_table('output00.txt')

success = 0
fail = 0
for x, y in zip(X_test, y_true):
    y_pred = naive_bayes.predict(x)
    if y_pred == y:
        success += 1
    else:
        fail += 1

print 'Success: ', success/(success + fail)
```

    Success:  1
    
