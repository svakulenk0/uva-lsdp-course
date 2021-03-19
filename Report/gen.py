import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from collections import Counter


# Cleans string from JSON leftovers and interpunction
def clean_text(text):
    text = re.sub("\'", "", text) 
    text = re.sub("[^a-zA-Z]"," ",text) 
    text = ' '.join(text.split()) 
    text = text.lower() 
    return text

# Removes stopwords 
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

#  Train OnevsRest model, choose between lr or svm
def train_MLR(xtrain, ytrain, xval, model='lr'):
    xtraincopy = xtrain.copy()
    xtraincopy.sort_indices()
    if model != 'lr' and model != 'svm':
        print('Wrong model type passed, choose for lr or svm. \n Training lr instead.')
        model = 'lr'
    elif model == 'lr':
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    elif model == 'svm':
        clf = OneVsRestClassifier(svm.SVC(C = 1, probability=True, kernel = 'linear', degree=3), n_jobs=-1)
    # we fit on a copy, to avoid writing to protected memory
    clf.fit(xtraincopy, ytrain)
    y_pred = clf.predict(xval)
    return y_pred, clf


# Make genre dataset usable.
def preprocess(meta):
    meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]
    
    plots = []
    genres = [] 
    movie_id = []
    plot = []
              
    with open("Datasets/MovieSummaries/plot_summaries.txt", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, dialect='excel-tab') 
        for row in tqdm(reader):
            plots.append(row)


    # Extract movie Ids and plot-summaries
    for i in plots:
        movie_id.append(i[0])
        plot.append(i[1])

    movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})
    # Change datatype of 'movie_id'
    meta['movie_id'] = meta['movie_id'].astype(str)

    # Merge meta with movies
    movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')
    
    # Extract genres
    for i in movies['genre']: 
        genres.append(list(json.loads(i).values())) 
        
    all_genres = sum(genres,[])
    movies['genre_new'] = genres
    all_genres = nltk.FreqDist(all_genres) 

    # Create dataframe
    all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                                  'Count': list(all_genres.values())})
    # Remove genres with too little occurences
    new_genres_df = all_genres_df[all_genres_df['Count'] > 1500]
    for i in range(len(movies)):
        movies['genre_new'][i] = [x for x in movies['genre_new'][i] if x in list(new_genres_df['Genre'])]
    
    # Remove movies with no genres 
    movies_new = movies[~(movies['genre_new'].str.len() == 0)]
    return movies_new, new_genres_df

def nw_acc(yval,ypred):
    score = 0
    for i,j in zip(yval,ypred):
        if np.dot(i , j) / len(i) * 100 > 0:
            score +=1
    return score/ len(yval)

def get_est(yval,ypred):
    test = []
    for i,j in zip(yval,ypred):
        test.append(np.dot(i , j) / len(i) * 100)
    return test

def norma(fp, genre_counts):
    norm = {}
    dicts = {}
    genre_counts.reset_index(drop=True, inplace=True)
    for i in range(len(genre_counts)):
        dicts[genre_counts["Genre"][i]] = genre_counts["Count"][i]
        
    for genre in fp.keys():
        temp = fp[genre] / dicts[genre] 
        norm[genre] = temp * 5
    
    return norm

def over_rep(est, multilabel_binarizer, yval, y_pred):
    fn = []
    fp = []
    for i in range(len(est)):
        if est[i] < 1:
            fn.append(multilabel_binarizer.inverse_transform(yval)[i])
            fp.append(multilabel_binarizer.inverse_transform(y_pred)[i])
    
    fn1 = list(sum(fn, ()))
    fp1 = list(sum(fp,()))
    return Counter(fn1), Counter(fp1)

def rescale(lst, val):
    tot = []
    for l in lst:
        nlist = []
        for item in l:
            if item >= val:
                nlist.append(int(1))
            else:
                nlist.append(int(0))
        tot.append(nlist)
    return np.array(tot)