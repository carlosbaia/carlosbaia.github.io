---
layout:     post
title:      Visualização e Análise de Dados
date:       2017-08-17 13:30:00
summary:    Visualização e Análise de Dados
thumbnail:  pie-chart
fb_image:   http://carlosbaia.com/img/visualizacao-e-analise-de-dados/cover.png

tags:
 - Machine Learning
 - Matplotlib
 - Seaborn
 - Data Visualization
---

Nesse artigo vamos falar sobre exploração de dados, pegaremos duas bases de dados (dataset) e veremos como gerar alguns gráficos com o objetivo de entender a estrutura dos dados e achar padrões.

As bibliotecas utilizadas para visualizar os dados serão:
<br>
[**Matplotlib**](http://matplotlib.org/){:target="_blank"}: é uma biblioteca utilizada para plotar os mais variados tipos de gráficos.
<br>
[**Seaborn**](https://seaborn.pydata.org/){:target="_blank"}: utiliza o Matplotlib como base e fornece funções simples de usar para criação de gráficos que seriam complexos apenas com o Matplotlib.

Instalação:
{% highlight python %}
pip install matplotlib seaborn
{% endhighlight %}

---

### Mercado financeiro  {#stocks}
Podemos facilmente plotar gráficos de ações de empresas na bolsa de valores. 

Vamos utilizar a biblioteca [pandas-datareader](https://github.com/pydata/pandas-datareader){:target="_blank"} para pegar informações sobre ações e preencher nosso dataset. Essa biblioteca não pega dados da Bovespa.

Vamos instalá-la com o seguinte comando:


{% highlight python %}
pip install pandas-datareader
{% endhighlight %}

Uma vez instalada, vamos aos imports padrões.

{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
{% endhighlight %}

A linha logo abaixo faz os gráficos aparecerem no Jupyter notebook, não é necessária caso não esteja utilizando ele.

{% highlight python %}
%matplotlib inline
{% endhighlight %}

Vamos pegar os valores das ações do Google da API do Google Finances de hoje até um ano atrás. A documentação do DataReader pode ser vista [aqui](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-google){:target="_blank"}.

{% highlight python %}
import pandas_datareader.data as pdr
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

df_goog = pdr.DataReader('GOOG', 'google', start_date, end_date)
df_goog.tail()
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/table1.png)

Uma vez que estamos com as informações em mãos, vamos aos gráficos.
Exibiremos os valores máximos e mínimos de cada dia de todo o período.

{% highlight python %}
df_goog[['High', 'Low']].plot(figsize=(15, 4), title='Google Stocks', grid=True)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_11_1.png)

Criaremos algumas novas colunas com base nos dados. Média móvel e retorno diário são métricas comumentes analisadas em ações.

{% highlight python %}
# Média movel de 14 dias do Fechamento
df_goog['MovingMean14'] = df_goog.Close.rolling(14).mean()
# Média movel de 30 dias do Fechamento
df_goog['MovingMean30'] = df_goog.Close.rolling(30).mean()
# Retorno diário percentual
df_goog['DailyReturn'] = df_goog.Close.pct_change()
df_goog.tail()
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/table2.png)

Vamos plotar o valor de fechamento e suas médias móveis para compararmos.

{% highlight python %}
columns = ['Close','MovingMean14', 'MovingMean30']
graph = df_goog[columns].plot(figsize=(15, 4), grid=True)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_15_0.png)

Caso queira salvar o gráfico em arquivo, basta guardar o retorno em uma variável como no exemplo anterior e chamar o método savefig do figure.


{% highlight python %}
graph.figure.savefig('graph.png')
{% endhighlight %}

Vamos ver como ficou o retorno diário das ações no decorrer do ano.

{% highlight python %}
df_goog.DailyReturn.plot(figsize=(15, 4), grid=True)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_19_1.png)

Ao plotar a distribuição desses dados, analisaremos mais facilmente se é uma ação com um bom retorno diário.

{% highlight python %}
fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))
sns.distplot(df_goog.DailyReturn.dropna(), bins=100, ax=ax1)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_21_1.png)

Quando já temos o figure, como é o caso acima, basta chamar o método savefig para salvar em disco.

{% highlight python %}
fig.savefig('graph2.png')
{% endhighlight %}

Com uma coluna que mostra somente ano e mês podemos agrupar os dados e gerar um boxplot para ver a distribuição dos dados em cada mês.

{% highlight python %}
fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))
df_goog['Month'] = df_goog.index.to_period('M')
sns.boxplot('Month', 'Close', data=df_goog, ax=ax1)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_25_1.png)


Transformaremos a data que é o index de nosso DataFrame em uma coluna normal para facilitar a criação de alguns gráficos, também criaremos uma coluna com a data em formato numérico.

{% highlight python %}
import matplotlib.dates as mdates

df_goog.reset_index(inplace=True)
df_goog['DateAsNumber'] = df_goog.Date.apply(mdates.date2num)
{% endhighlight %}

Com o Matplotlib conseguimos plotar um gráfico do tipo candlestick, que é muito comum na analise de ações. Usaremos os últimos 15 dias para o gráfico ficar com um tamanho legível.

{% highlight python %}
from matplotlib.finance import candlestick_ohlc

def plot_candle_stick(df):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))
    candlestick_ohlc(ax1, df.values, width=.6, colorup='g', colordown='r')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price')

columns = ['DateAsNumber', 'Open', 'High', 'Low', 'Close', 'Volume']
plot_candle_stick(df_goog[columns].tail(15))
{% endhighlight %}


![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_29_0.png)

Com uma curva de tendência para o valor de fechamento das ações, podemos ter uma projeção para os próximos valores. Uma regressão linear será usada para criar essa curva.

{% highlight python %}
lm = sns.lmplot('DateAsNumber', 'Close', data=df_goog, aspect=2.5, order=3)
lm.ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
lm.ax.set_xlabel('Date')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_31_1.png)

Os gráficos do Seaborn retornam um objeto onde tem o figure que usamos para salvar em disco:

{% highlight python %}
lm.fig.savefig('graph3.png')
{% endhighlight %}

<br>

#### Múltiplas ações

Podemos plotar gráficos de múltiplas ações simultaneamente, o que é muito útil para fins comparativos. Vamos pegar um período de 5 anos.

{% highlight python %}
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

stock_names = ['GOOG', 'MSFT', 'AMZN', 'AAPL','TSLA', 'XOM', 'GE', 'EBAY']
df_stocks = pdr.DataReader(stock_names, 'google', start_date, end_date)
df_stocks.Close.tail()
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/table3.png)

{% highlight python %}
df_stocks.Close.plot(figsize=(15, 4), grid=True, colormap='Set1')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_37_1.png)

No código acima usamos o *Set1* no **colormap**, é a paleta de cor que usamos para colorir o gráfico. No caso do Seaborn a paleta pode ser passada pelas variáveis **cmap** ou **palette**. As diversas paletas existentes podem ser vistas [aqui](http://matplotlib.org/1.2.1/_images/show_colormaps.png){:target="_blank"}. Quando o parâmetro for apenas **color**, usamos cores puras (red, blue, black etc) ao invés de paletas.

Geraremos um gráfico que mostra a correlação entre as ações.

{% highlight python %}
def plot_corr(corr):
    # Cortaremos a metade de cima pois é o espelho da metade de baixo
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True

    sns.heatmap(corr, mask=mask, cmap='RdBu', square=True, linewidths=.5)

# Calculando a correlação
corr = df_stocks.Close.corr() 
plot_corr(corr)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_39_0.png)

Vemos no exemplo que empresas de tecnologia (GOOG, MSFT, AMZN, AAPL e TSLA) possuem alta correlação entre sim, enquanto empresas de outros ramos, como energia (GE) e petróleo (XOM) possuem correlação baixa com essas empresas de tecnologia. Totalmente azul (1) é correlação máxima, totalmente vermelho (-1) é correlação inversa e branco (0) é sem qualquer correlação. Vemos que as ações possuem correlação máxima com elas mesmas e estão azul.

---

### Titanic  {#titanic}

É um famoso problema do site de competições Kaggle. É uma base de dados que contém diversas informações sobre os passageiros do Titanic, baseado nessas informações devemos criar um modelo capaz de dizer se determinada pessoa morreu ou não no naufrágio. Mais informações e a base de dados para download podem ser encontradas [aqui](https://www.kaggle.com/c/titanic){:target="_blank"}.

Temos diversas informações nesse dataset, como idade, sexo, cabine, cidade de embarque, número de acompanhantes, valor pago, classe e se a pessoa morreu ou não no acidente. Para uma parte dessas pessoas o site não fornece essa última informação, dessa forma o site consegue validar se o seu modelo é capaz de prever corretamente algo que ele não conhece. Nesse primeiro artigo vamos apenas visualizar e analisar os dados, em um próximo treinaremos um modelo com ele.

{% highlight python %}
df = pd.read_csv('train.csv')
df.head()
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/table4.png)

Criaremos um gráfico mostrando a distribuição do sexo masculino e feminino, podemos usar a função *plot* do próprio Pandas ou o *factorplot* do Seaborn.

{% highlight python %}
# Com Pandas
df.Sex.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')  # Para deixar o gráfico redondo

# Com Seaborn
sns.factorplot('Sex', data=df, kind='count')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_44_1.png)

Faremos o mesmo para a classe onde o passageiro se encontra.

{% highlight python %}
df.Pclass.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal');

sns.factorplot('Pclass',data=df, kind='count')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_46_1.png)

Vejamos como ficou a distribuição entre sobreviventes e não sobreviventes.

{% highlight python %}
# Para trocar os valores numéricos pelos nomes nos gráficos
survived_map = {0: 'Died', 1: 'Survived'}

df.Survived.map(survived_map).value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal');

sns.factorplot('Survived',data=df, kind='count').set_xticklabels(survived_map.values())
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_48_1.png)

Podemos usar o campo **hue** para dividir as barras por categorias.

{% highlight python %}
# Distribuição do sexo dividido por classe.
sns.factorplot('Sex', data=df, hue='Pclass', kind='count')
# Distribuição da classe dividida por sexo.
sns.factorplot('Pclass', data=df, hue='Sex', kind='count')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_50_1.png)

Geraremos uma coluna dizendo se é adulto ou não para vermos a distribuição de adultos e crianças em cada sexo.

{% highlight python %}
df['is_adult'] = df.Age.apply(lambda age: age >= 18)
sns.factorplot('Sex', data=df, hue='is_adult', kind='count')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_52_1.png)

Com um histograma, podemos ver a distribuição da idade dos passageiros. Vemos que a maioria está por volta dos 20, 30 anos.

{% highlight python %}
df.Age.hist(bins=int(df.Age.max()), figsize=(15, 4))
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_54_1.png)

Olhando a cidade de embarque dos passageiros, vemos que a maioria embarcou em *Southampton*.

{% highlight python %}
city_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

df.Embarked.map(city_map).value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal');
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_56_0.png)

Criaremos uma nova coluna agrupando as pessoas em homem, mulher, criança e idoso baseado no sexo e na idade.

{% highlight python %}
def get_description(row):
    if row.Age < 18:
        return 'child'
    elif row.Age >= 65:
        return 'elderly'
    return 'man' if row.Sex == 'male' else 'woman'

df['Description'] = df.apply(get_description, axis=1)
df.Description.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_58_1.png)

A taxa de sobrevivência desses novos grupos ficam assim:

{% highlight python %}
sns.factorplot('Survived', data=df, hue='Description', kind='count').set_xticklabels(survived_map.values())
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_60_1.png)

O gráfico mostra que a taxa de mortalidade é muito maior entre os homens. Isso provavelmente é devido à política de "mulheres e crianças na frente".

Com as novas colunas abaixo veremos quem viaja acompanhado e com quantos membros da familía está.

{% highlight python %}
# Coluna membros da família = irmãos/parceiros + país/filhos.
df['FamilyMember'] = df.SibSp + df.Parch
# Coluna dizendo se a pessoa está sozinho ou não.
df['IsAlone'] = df.FamilyMember == 0

df.IsAlone.value_counts().plot(kind='pie', autopct='%.2f%%')
plt.axis('equal')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_62_1.png)

Vemos que a maior parte das pessoas viajavam sozinhas.

{% highlight python %}
sns.factorplot('Survived', data=df, hue='IsAlone', kind='count').set_xticklabels(survived_map.values())
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_64_1.png)

Também vemos que a taxa de mortalidade foi bem maior entre os que viajavam sozinhos.

{% highlight python %}
sns.factorplot('Survived',data=df, hue='Pclass', kind='count', aspect=3, order=survived_map.keys()).set_xticklabels(survived_map.values())
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_66_1.png)

Entre as classes, vemos que a primeira teve mais sobreviventes do que mortos, a segunda foi equilibrada e a terceira teve um número bem maior de morto. Isso provavelmente é devido à posição das cabines com relação ao ponto de impacto do icerberg.

Cruzaremos alguns dados com à cidade de embarque.

{% highlight python %}
sns.factorplot('Embarked',data=df, hue='Pclass', kind='count', aspect=3, order=city_map.keys()).set_xticklabels(city_map.values())

sns.factorplot('Embarked',data=df, hue='IsAlone', kind='count', aspect=3, order=city_map.keys()).set_xticklabels(city_map.values())

sns.factorplot('Embarked',data=df, hue='Description', kind='count', aspect=3, order=city_map.keys()).set_xticklabels(city_map.values())
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_69_1.png)

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_69_2.png)

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_69_3.png)

A cidade de Southampton é na sua maioria composta de pessoas da terceira classe, sozinhas e homens. Pelo perfil é bem provável que tenha alguma indústria nessa cidade e sejam pessoas voltando do trabalho.

Vamos ver quem viajou sozinho.

{% highlight python %}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

df[df.IsAlone].Description.value_counts().plot(kind='pie', autopct='%.2f%%', ax=ax1, title='Alone people')
ax1.axis('equal')
df[~df.IsAlone].Description.value_counts().plot(kind='pie', autopct='%.2f%%', ax=ax2, title='People with Family')
ax2.axis('equal')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_72_1.png)

A maioria dos viajantes sozinhos são homens e quase não temos crianças sozinhas. Já nas pessoas com família, temos uma incidência maior de mulheres e muitas crianças. Temos poucos idosos no navio, tanto sozinhos, quanto acompanhados.

Com um gráfico de linha veremos a distribuição dos dados.

{% highlight python %}
def plot_line_graph(x, hue):
    fig = sns.FacetGrid(df, hue=hue, aspect=4)
    fig.map(sns.kdeplot, x, shade=True)
    fig.set(xlim=(0, df[x].max()))
    fig.add_legend()
    return fig

plot_line_graph('Fare', 'Pclass')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_75_1.png)

Distribuindo o valor das taxas de embarque de acordo com a classe, vemos que as maiores são as da primeira classe.

{% highlight python %}
fig = plot_line_graph('Age', 'Sex')
fig.ax.axvline(18, color='r')
fig.ax.axvline(65, color='gray')
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_77_1.png)

Plotando as idades de acordo com o sexo, vemos que a distribuição de idade entre homens e mulheres é bem próxima.

Por último, criaremos gráficos que mostram a taxa de sobrevivência baseado em algumas das características que temos.

{% highlight python %}
sns.factorplot('Pclass', 'Survived', data=df)
sns.factorplot('Description', 'Survived', data=df)
sns.factorplot('Embarked', 'Survived', data=df, order=city_map.keys()).set_xticklabels(city_map.values())
sns.factorplot('IsAlone', 'Survived', data=df)
{% endhighlight %}

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_80_1.png)

![image]({{ site.baseurl }}/img/visualizacao-e-analise-de-dados/output_80_3.png)

O grupo que mais sobreviveu possui as seguintes características: está na primeira classe, é mulher, embarcou em Cherbourg e não viajava sozinha.

Diversos outros gráficos podem ser encontrados na documentação de [**Matplotlib**](https://matplotlib.org/api/index.html){:target="_blank"} e [**Seaborn**](https://seaborn.pydata.org/api.html){:target="_blank"}, além de existirem dezenas de outras bibliotecas para esse fim.

O objetivo desse artigo foi mostrar como plotar os dados para melhor entendê-los e através disso criar melhores modelos de Machine Learning.

O Jupyter notebook com todo o código pode ser visto [aqui](https://github.com/carlosbaia/carlosbaia.github.io/blob/master/notebook/visualizacao-e-analise-de-dados.ipynb){:target="_blank"}.
