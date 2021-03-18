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


#Lectura de la data 
X_train = pd.read_csv("DataKaggle/train.csv - Hoja 1.tsv",quoting = 3, delimiter = "\t", header= 0)
X_test = pd.read_csv("DataKaggle/test.csv - Hoja 1.tsv", quoting = 3, delimiter = "\t", header = 0)

#print(X_train['comment_es'][0][:600])

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

tv = TfidfVectorizer(ngram_range = (1,3),
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

# Entrenamiento 

X_train['freq_word'] = X_train['clean'].apply(lambda x: len(str(x).split()))
X_train['unique_freq_word'] = X_train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
X_test['freq_word'] = X_test['clean'].apply(lambda x: len(str(x).split()))
X_test['unique_freq_word'] = X_test['clean'].apply(lambda x: len(set(str(x).split())))  


kfold = StratifiedKFold( n_splits = 5, random_state = 42 , shuffle=True )

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