---
layout:     post
title:      Decision Tree e Random Forest
date:       2016-07-20 12:00:00
summary:    Implementando Decision Tree e Random Forest com o SkLearn
thumbnail:  book
fb_image:   http://carlosbaia.com/img/decision-tree-e-random-forest/cover.png
tags:
 - Machine Learning
 - SkLearn
 - Decision Tree
 - Random Forest
 - Cross Validation
 - Feature Importance
 - Grid Search
---

Nesse artigo vamos fazer nossa primeira implementação prática de um algoritmo de Machine Learning.
Pegaremos uma base de dados real, faremos algumas análises nos dados e logo após implementaremos nosso primeiro algoritmo.
Usaremos um algoritmo de [**classificação**]({{ site.baseurl }}/2016/07/17/introducao-ao-machine-learning/#supervised){:target="_blank"} para atingir nosso objetivo.

### Ambiente de desenvolvimento
Antes de desenvolver nosso primeiro código, vamos configurar nosso ambiente de desenvolvimento.
A linguagem de programação utilizada será o [**Python**](https://www.python.org/){:target="_blank"} e as principais bibliotecas utilizadas serão:

- [**Numpy**](http://www.numpy.org/){:target="_blank"}: Biblioteca matemática muito poderosa, facilita muito o trabalho com arrays e tem diversas funções de algebra, estática e computação científica no geral. 
É implementada em C para garantir alta performance, o que é muito importante quando se trabalha com uma grande quantidade de dados.
- [**Pandas**](http://pandas.pydata.org/){:target="_blank"}: Muito útil para estruturar os dados, ajuda muito na analise e na manipulação de grande quantidade de dados.
Também é implementada em C para garantir alta performance.
- [**Matplotlib**](http://matplotlib.org/){:target="_blank"}: Utilizada para plotar gráficos, o que uma ótima ferramenta na análise dos dados.
- [**SciKit Learn** ou **SkLearn**](http://scikit-learn.org){:target="_blank"}: Possui diversos algoritmos de Machine Learning. Será a principal biblioteca que utilizaremos.

Para desenvolver o código é possivel usar sua IDE de preferência, caso não tenha uma, recomendo o uso do [**Jupyter**](http://jupyter.org/){:target="_blank"}
para as analises iniciais da base e os primeiros testes. Quando a complexidade do código aumentar é só migrar para o [**PyCharm**](https://www.jetbrains.com/pycharm/download/){:target="_blank"} e continuar a desenvolver. 

Não é necessário instalar o Python e todas essas bibliotecas uma a uma, o [**Anaconda**](https://www.continuum.io/downloads){:target="_blank"} é uma plataforma para data science que já inclui tudo isso.
É só instalar e começar a desenvolver.
Todas as ferramentas citada funcionam em Windows, Linux e Mac.

### Base de dados
Um bom local para encontrar dados para praticar, é o [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/ml/datasets.html){:target="_blank"}.
Lá você encontra base para diversas áreas, algoritmos e tamanhos. A maioria dos problemas são bem detalhados, com referências e detalhes de como foram extraídas as features iniciais.

A base escolhida para o nosso exemplo foi a [**Iris Data Set**](https://archive.ics.uci.edu/ml/datasets/Iris){:target="_blank"}.
O problema proposto é através das característas fornecidas de uma flor descobrir de qual das flores abaixo que se trata.
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/iris-flowers.png)

Os dados estão em formato CSV e possui 150 entradas:
{% highlight python %}
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
...
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
...
{% endhighlight %}

Cada uma das entradas possui 5 colunas com a seguinte descrição:
{% highlight c %}
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
  - Iris Setosa
  - Iris Versicolour
  - Iris Virginica
{% endhighlight %}

Petal (pétala) e sepal (sépala) são partes da flor, são fornecidas as dimensões em centímetros dessas partes e 
com isso devemos classificar a flor entre Iris Setosa, Versicolour ou Virginica.
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/iris_petal_sepal.png)

### Analisando os dados
Baixe a base de dados nesse [link](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data){:target="_blank"} e vamos utilizar o Pandas para carregar os dados.<br>
O Pandas por padrão pega a primeira linha como nome das colunas, porém ao abrir a base vemos que esses nomes não estão presentes nessa base.<br>
Com base na descrição fornecida, vamos passar esses nomes na leitura dos dados.

{% highlight python %}
import pandas as pd
names = ['SepalLength', 'SepalWidth',
         'PetalLength', 'PetalWidth',
         'Class']
df = pd.read_csv('iris.data', names=names)
{% endhighlight %} 

O objeto df é um [DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html){:target="_blank"}, nele está os dados carregados e diversas funções muito poderosas para manipular os dados.
Recomendo a leitura da documentação para entender tudo que ele pode fazer.

Com os códigos abaixo começamos a ver o tamanho e formato dos dados.
{% highlight python %}
print("Linhas: %d, Colunas: %d" % (len(df), len(df.columns)))
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output: Linhas: 150, Colunas: 5*

A função *head* nos mostra por padrão as primeiras 5 linhas do DataFrame. Somos capazes de ver ai o nome de nossas features e a classificação (Class) da flor.
{% highlight python %}
df.head()
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output:*
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/df_head.png)

Vamos ver quantas flores de cada categória existe na nossa base.
{% highlight python %}
df['Class'].value_counts()
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Iris-versicolor    50<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Iris-setosa        50<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Iris-virginica     50<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name: Class, dtype: int64*

Vemos que a base está bem distribuida, com a mesma quantidade de dados para todas as classes.
Isso é um informação importante na hora de escolher o tipo [**métrica (score)**](http://scikit-learn.org/stable/modules/model_evaluation.html){:target="_blank"} usaremos para avaliar o resultado do nosso modelo.
Podemos também plotar essa distruição com o Matplotlib, para visualizar melhor os dados.
{% highlight python %}
%matplotlib inline  # Para os graficos aparecerem no jupyter
import matplotlib.pyplot as plt
df['Class'].value_counts().plot(kind='pie');
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output:*
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/df_classes.png)

### Criando features
A criação de [**features**]({{ site.baseurl }}/2016/07/17/introducao-ao-machine-learning/#feature){:target="_blank"} é um dos passos mais importantes do ML,
quanto melhor forem as features, mais facilmente o algoritmo irá achar padrões nos dados.
<br><br>
Uma forma de gerar features é aplicando matemática e estatística nos dados.
O Pandas é muito poderoso na manipulação dos dados, o que facilita muito a geração de novas features.
<br>
Um exemplo que podemos ver é fazer operações como *, +, > ele faz isso para cada linha da coluna e retorna uma nova coluna com os resultados:

![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/pandas_sample.png)

{% highlight python %}
# Como temos largura e comprimento, podemos criar uma feature de area
df['SepalArea'] = df['SepalLength'] * df['SepalWidth']
df['PetalArea'] = df['PetalLength'] * df['PetalWidth']

# Vamos tirar a media de cada feature e criar uma feature boleana
# que marca linha por linha se esses valores estao acima da media.
df['SepalLengthAboveMean'] = df['SepalLength'] > df['SepalLength'].mean()
df['SepalWidthAboveMean'] = df['SepalWidth'] > df['SepalWidth'].mean()

df['PetalLengthAboveMean'] = df['PetalLength'] > df['PetalLength'].mean()
df['PetalWidthAboveMean'] = df['PetalWidth'] > df['PetalWidth'].mean()
{% endhighlight %}
Vamos ver os dados novamente, além das colunas que já vimos anteriormente, podemos ver também as novas colunas abaixo.
{% highlight python %}
df.head()
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output:*
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/df_head_features.png)

Além do Pandas, o Numpy e o próprio possui uma vasta quantidade de funções que podem ajudar na manipulação de dados e criação de features.
Com o Matplotlib também é possivel plotar diferentes tipos de gráficos, que ajudar a entender melhor os dados e dar ideia para criação de features.
<br>
Recomendo dar uma olhada na documentação dessas bibliotecas e procurar exemplos de uso na internet para ter ideias e entender o que cada uma pode te oferecer.

### Treinamento
{% highlight python %}
y = df['Class'].values
df.drop('Class', axis=1, inplace=True)
X = df.values
{% endhighlight %}


{% highlight python %}
# Iris-setosa
sample1 = [5.0, 3.5, 1.3, 0.2, 17.8, 0.2, False, True, False, False]
# Iris-versicolor
sample2 = [5.0, 2.0, 3.5, 1.0, 10.0, 3.5, False, False, False, False]
# Iris-virginica
sample3 = [5.9, 3.0, 5.0, 1.8, 17.7, 9.1, True, False, True, True]
{% endhighlight %}

<br>

#### Decision Tree
Também conhecido como árvore de decisão, é um al

{% highlight python %}
from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier(random_state=1986,
                                       max_depth=3)
classifier_dt.fit(X, y)

classifier_dt.predict([sample1, sample2, sample3])
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output: array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/decision-tree.png)

#### Random Forest
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/random_forest.png)
{% highlight python %}
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=1986,
                                       n_estimators=50,
                                       max_depth=10,
                                       n_jobs=-1)
classifier_rf.fit(X, y)

classifier_rf.predict([sample1, sample2, sample3])
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output: array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

### Cross Validation
![image]({{ site.baseurl }}/img/decision-tree-e-random-forest/kfolds.png)

{% highlight python %}
from sklearn.cross_validation import cross_val_score

scores_dt = cross_val_score(classifier_dt, X, y, scoring='accuracy', cv=3)
scores_rf = cross_val_score(classifier_rf, X, y, scoring='accuracy', cv=3)

print("DT: %f, RF %f" % (scores_dt.mean(), scores_rf.mean()))
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output: DT: 0.966912, RF 0.967320*

### Feature Importance

{% highlight python %}
features_importance = zip(classifier_rf.feature_importances_, df.columns)
for importance, feature in sorted(features_importance, reverse=True):
    print("%s: %f%%" % (feature, importance*100))
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature ranking:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PetalArea: 30.255950%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PetalLength: 27.133493%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PetalWidth: 24.840767%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PetalLengthAboveMean: 5.582292%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SepalLength: 3.817337%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PetalWidthAboveMean: 2.703280%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SepalArea: 2.258624%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SepalWidth: 1.548223%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SepalLengthAboveMean: 1.371979%
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SepalWidthAboveMean: 0.488055%

### Grid Search

{% highlight python %}
from sklearn.grid_search import GridSearchCV

param_grid = {
            "criterion": ['entropy', 'gini'],
            "n_estimators": [25, 50, 75],
            "bootstrap": [False, True],
            "max_depth": [3, 5, 10],
            "max_features": ['auto', 0.1, 0.2, 0.3]
}
grid_search = GridSearchCV(classifier_rf, param_grid, scoring="accuracy")
grid_search.fit(X, y)

classifier_rf = grid_search.best_estimator_ 
grid_search.best_params_, grid_search.best_score_
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Output:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;({'bootstrap': True,
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  'criterion': 'gini',
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  'max_depth': 3,
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  'max_features': 0.1,
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  'n_estimators': 25},
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.97333333333333338)*

