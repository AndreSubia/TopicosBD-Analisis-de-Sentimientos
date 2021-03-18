#Librerias para la manipulacion de datos
import pandas as pd
import numpy as np

#Procesamiento de texto
from bs4 import BeautifulSoup
import re
import nltk

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words

#Visualizacion
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Modelos
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression

# Read the data 
X_train = pd.read_csv("DataKaggle/train.csv - Hoja 1.tsv",quoting = 3, delimiter = "\t", header= 0)
X_test = pd.read_csv("DataKaggle/test.csv - Hoja 1.tsv", quoting = 3, delimiter = "\t", header = 0)

print(X_train['comment_es'][0][:600])

print('Training set dimension:',X_train.shape)
print('Test set dimension:',X_test.shape)

def prep(comment):
    
    # Remover tags de HTML.
    comment = BeautifulSoup(comment,'html.parser').get_text()
    
    # Remover todo lo que no sean letras.
    comment = re.sub("[^a-zA-Z]", " ", comment)
    
    # Convertir el texto en minusculas.
    comment = comment.lower()
    #print(comment)
    
    # Tokenizacion de las palabras.
    token = nltk.word_tokenize(comment)
    
    # Stemming
    comment = [nltk.stem.SnowballStemmer('spanish').stem(w) for w in token]
    
    # Concatenación de las palabras separadas por un espacio.
    return " ".join(comment)

#Aplicamos la funcion de preprocesamiento para cada elemento de la dataset.

X_train['clean'] = X_train['comment_es'].apply(prep)
X_test['clean'] = X_test['comment_es'].apply(prep)

tv = TfidfVectorizer(
                    ngram_range = (1,3),
                    sublinear_tf = True,
                    max_features = 40000)


train_tv = tv.fit_transform(X_train['clean'])
test_tv = tv.transform(X_test['clean'])

# Creamos la lista del vocabulario para que sea vectorizada.

vocab = tv.get_feature_names()
print(vocab[:5])

print("Vocabulary length:", len(vocab))


dist = np.sum(train_tv, axis=0)
checking = pd.DataFrame(dist,columns = vocab)

print('Training dim:',train_tv.shape, 'Test dim:', test_tv.shape)

def cloud(name, data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(name+'.png')
    plt.show()

cloud('cloud_train1',' '.join(X_train['clean']))
#cloud('cloud_test',' '.join(X_test['clean']))

X_train['freq_word'] = X_train['clean'].apply(lambda x: len(str(x).split()))
X_train['unique_freq_word'] = X_train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
X_test['freq_word'] = X_test['clean'].apply(lambda x: len(str(x).split()))
X_test['unique_freq_word'] = X_test['clean'].apply(lambda x: len(set(str(x).split())))  

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(X_train['freq_word'], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(X_train['freq_word'])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribucion de la Frecuencia de las Palabras")
axes[0].axvline(X_train['freq_word'].median(), linestyle='dashed')
print("media de la frecuencia de palabras: ", X_train['freq_word'].median())

sns.distplot(X_train['unique_freq_word'], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(X_train['unique_freq_word'])
axes[1].set_title("Distribucion de la Frecuencia de las Palabras Unicas")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(X_train['unique_freq_word'].median(), linestyle='dashed')
print("media de la frecuencia de palabras unicas: ", X_train['unique_freq_word'].median())

plt.savefig('dist.png')


kfold = StratifiedKFold( n_splits = 5, random_state = 42 , shuffle=True )



lr = LogisticRegression(random_state = 2018)

lr2_param = {
    'penalty':['l2'],
    'dual':[False],
    'C':[6],
    'class_weight':[{1:1}]
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(train_tv, X_train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_

print(lr_CV.best_score_)


sv = LinearSVC(random_state= 42)

param_grid2 = {
    'loss':['squared_hinge'],
    'class_weight':[{1:4}],
    'C': [0.2]
}


gs_sv = GridSearchCV(sv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')

#print(train_tv)

gs_sv.fit(train_tv, X_train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)