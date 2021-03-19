import pandas as pd
import numpy as np
import csv

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import nltk, sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes, svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from lime import lime_text
from lime.lime_text import LimeTextExplainer

from collections import Counter

from nltk.corpus import stopwords

# Train Support Vector Machine
def train_svm(df_train, df_test):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(df_train.utterance)
    dev_vectors = vectorizer.transform(df_test.utterance)
    # Fit Support Vector Machine model 
    tester = svm.SVC(C = 1, probability=True, kernel = 'linear', degree=3)
    tester.fit(train_vectors, df_train.Character_label)
    # Predict intent labels for test set with Support Vector Machine
    pred_svm = tester.predict(dev_vectors)
    # Pipeline for Lime
    svmp = make_pipeline(vectorizer, tester)
    return svmp, pred_svm

# Train Logistic Regression
def train_lr(df_train, df_test):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(df_train.utterance)
    dev_vectors = vectorizer.transform(df_test.utterance)
    # Fit Logistic Regression model 
    lr = LogisticRegression(penalty='l2')
    lr.fit(train_vectors, df_train.Character_label)
    # Predict intent labels for test set with Logistic Regression
    pred_lr = lr.predict(dev_vectors)
    # Pipeline for Lime
    lrp = make_pipeline(vectorizer, lr)
    return lrp, pred_lr

# Train Naive Bayes
def train_nb(df_train, df_test):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(df_train.utterance)
    dev_vectors = vectorizer.transform(df_test.utterance)
    # Fit Naive Bayes model
    nb = naive_bayes.MultinomialNB()
    nb.fit(train_vectors, df_train.Character_label)
    # Predict intent labels for test set with Naive Bayes
    pred_nb = nb.predict(dev_vectors)
    # Pipeline for Lime
    nbp = make_pipeline(vectorizer, nb)
    return nbp, pred_nb

# Train Decision Tree
def train_dt(df_train, df_test):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(df_train.utterance)
    dev_vectors = vectorizer.transform(df_test.utterance)
    # Fit Decision Tree model
    dt = sklearn.tree.DecisionTreeClassifier(random_state=0)
    dt.fit(train_vectors, df_train.Character_label)
    # Predict offensive labels for validation set with DT
    pred_dt = dt.predict(dev_vectors)
    # Pipeline for Lime
    dtp = make_pipeline(vectorizer, dt)
    return dtp, pred_dt

# Train Nearest Neighbours
def train_nn(df_train, df_test):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(df_train.utterance)
    dev_vectors = vectorizer.transform(df_test.utterance)
    # Fit Nearest Neighbour model
    nn = KNeighborsClassifier()
    nn.fit(train_vectors, df_train.Character_label)
    # Predict offensive labels for validation set with Nearest Neighbour
    pred_nn = nn.predict(dev_vectors)
    # Pipeline for Lime
    nnp = make_pipeline(vectorizer, nn)
    return nnp, pred_nn
    
# Lime explainer  plotter  
def explain_index_2(idx, classifier, n_features, dataset, explainer) :
    print(dataset['utterance'][idx])
    exp = explainer.explain_instance(dataset['utterance'][idx], classifier.predict_proba, num_features=n_features)
    print('Document id: %d' % idx)
    print('Probability(good sentiment) =', classifier.predict_proba([dataset['utterance'][idx]])[0,1])
    print('True class: %s' % dataset.Character_label[idx])
    print("Speaker: %s" % dataset.Speaker[idx])
    
    # Plot figures     
    figure = exp.as_pyplot_figure();
    in_notebook = exp.show_in_notebook(text=False)

    return figure, in_notebook

def conf_matrix(df_test, pred):
    print(len(df_test))
    matrix = [0,0,0,0]
    for i in range(len(df_test)):
        if df_test.Character_label[i] == pred[i]:
            if pred[i] == 1:
                matrix[3] += 1
            else:
                matrix[0] += 1
        else:
            if pred[i] == 1:
                matrix[1] += 1
            else:
                matrix[2] +=1
    return print('True positives:',matrix[3],'False positives:',matrix[1],'True negatives:',matrix[0],'False negatives:',matrix[2])
