#############################
# THIS FILE IS MADE BY:
# SHELBY JHORAI (ID:11226374)
#
# CONTENT:
# - Save and Load
# - Imports
# - Preprocessing
# - Classification model
# - Evaluation
# - Annotation
# - Results
#############################


##########
# IMPORTS
##########


import os
import re
import timeit
import string
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import walk
from collections import defaultdict
from collections import OrderedDict

import nltk
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


################
# SAVE AND LOAD
################


def save_variables(saved_path, unique_emotions, v_size, matrix):
    with open(saved_path+'unique_emotions.txt', 'wb') as filehandle:
        pickle.dump(unique_emotions, filehandle)
    with open(saved_path+'v_size.txt', 'wb') as filehandle:
        pickle.dump(v_size, filehandle)
    with open(saved_path+'matrix.txt', 'wb') as filehandle:
        pickle.dump(matrix, filehandle)


def save_x_y(saved_path, X_train, X_test, y_train, y_test):
    with open(saved_path+'X_train.txt', 'wb') as filehandle:
        pickle.dump(X_train, filehandle)
    with open(saved_path+'X_test.txt', 'wb') as filehandle:
        pickle.dump(X_test, filehandle)
    with open(saved_path+'y_train.txt', 'wb') as filehandle:
        pickle.dump(y_train, filehandle)
    with open(saved_path+'y_test.txt', 'wb') as filehandle:
        pickle.dump(y_test, filehandle)


def save_vectors(saved_path, C_vec, M_vec):
    with open(saved_path+'C_vec.txt', 'wb') as filehandle:
        pickle.dump(C_vec, filehandle)
    with open(saved_path+'M_vec.txt', 'wb') as filehandle:
        pickle.dump(M_vec, filehandle)


def save_dfs(saved_path, COVID_df, MED_df):
    with open(saved_path+'COVID_df.txt', 'wb') as filehandle:
        pickle.dump(COVID_df, filehandle)
    with open(saved_path+'MED_df.txt', 'wb') as filehandle:
        pickle.dump(MED_df, filehandle)


def load_variables(saved_path):
    with open(saved_path+'unique_emotions.txt', 'rb') as filehandle:
        unique_emotions = pickle.load(filehandle)
    with open(saved_path+'v_size.txt', 'rb') as filehandle:
        v_size = pickle.load(filehandle)
    with open(saved_path+'matrix.txt', 'rb') as filehandle:
        matrix = pickle.load(filehandle)

    return unique_emotions, v_size, matrix


def load_x_y(saved_path):
    with open(saved_path+'X_train.txt', 'rb') as filehandle:
        X_train = pickle.load(filehandle)
    with open(saved_path+'X_test.txt', 'rb') as filehandle:
        X_test = pickle.load(filehandle)

    with open(saved_path+'y_train.txt', 'rb') as filehandle:
        y_train = pickle.load(filehandle)
    with open(saved_path+'y_test.txt', 'rb') as filehandle:
        y_test = pickle.load(filehandle)

    return X_train, X_test, y_train, y_test


def load_vectors(saved_path):
    with open(saved_path+'C_vec.txt', 'rb') as filehandle:
        C_vec = pickle.load(filehandle)
    with open(saved_path+'M_vec.txt', 'rb') as filehandle:
        M_vec = pickle.load(filehandle)

    return C_vec, M_vec


def load_dfs(saved_path):
    with open(saved_path+'COVID_df.txt', 'rb') as filehandle:
        COVID_df = pickle.load(filehandle)
    with open(saved_path+'MED_df.txt', 'rb') as filehandle:
        MED_df = pickle.load(filehandle)
    return COVID_df, MED_df


def load_trained_model(path, name):
    return load_model(path+name)


def load_history(saved_path, name):
    with open(saved_path+name, 'rb') as filehandle:
        history = pickle.load(filehandle)
    return history


################
# PREPROCESSING
################


def split_on_dialogue(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        f.close()
    i = 0
    j = 0
    dialogue_i = 0
    convo = []
    conversations = []
    for line in lines:
        i += 1
        tokens = word_tokenize(line)
        if line[:8] == 'Dialogue':
            dialogue_i = i+1
        if i == dialogue_i+j:
            convo.append(line)
            j += 1
            if len(tokens) == 0:
                conversations.append(convo)
                convo = []
                j = 0
                continue
    return conversations


def create_emotions(emotions_path):
    emotions = dict()
    unique_emotions = []
    lemmatizer = WordNetLemmatizer()
    _, _, files = next(walk(emotions_path))

    for file in files:
        emotion = file.replace('-scores.txt', '')
        unique_emotions.append(emotion)

        with open(emotions_path+file, 'r') as f:
            for line in f:
                word, p = line.split('\t')
                if float(p) > 0.6:
                    word = lemmatizer.lemmatize(word)
                    emotions[word] = emotion
    return unique_emotions, emotions


def remove_noise(token):
    token = re.sub(r"\\n", "", token)
    token = re.sub(r"\n", "", token)
    token = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                   r'(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    token = re.sub(r"(@[A-Za-z0-9_]+)", "", token)
    token = re.sub(r"-", " ", token)
    token = token.lower().translate(str.maketrans('', '', string.punctuation))
    return token


def lemmatize(token, tag):
    lemmatizer = WordNetLemmatizer()
    pos = False
    if tag.startswith("NN"):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    if pos:
        return lemmatizer.lemmatize(token, pos)
    else:
        return ''


def clean_text(raw_text):
    cleaned = []

    stop_words = set(stopwords.words('english')) | set(['http', 'patient', 'doctor'])
    tokens = word_tokenize(str(raw_text), "english")

    for token, tag in pos_tag(tokens):

        token = remove_noise(token)
        token = lemmatize(token, tag)

        if len(token) > 2 and len(token) < 20 and token not in stop_words:
            cleaned.append(token)
    return cleaned


def create_label(text, emotions):
    return [emotions[token] for token in text if token in emotions.keys()]


def create_df(labelled):
    df = pd.DataFrame(columns=['text', 'labels'])
    for text, label in labelled:
        df = df.append({'text': text, 'labels': label}, ignore_index=True)
    return df


def binarizer(df):
    # Binarise labels
    mlb = MultiLabelBinarizer()
    result = mlb.fit_transform(df['labels'])
    new_df = pd.concat([df['text'],
                        pd.DataFrame(result, columns=list(mlb.classes_))],
                       axis=1)
    return new_df


def append_dfs(df1, df2):
    if len(df1) < len(df2):
        df2 = df2.sample(len(df1))
    elif len(df2) < len(df1):
        df1 = df1.sample(len(df2))
    return pd.concat([df1, df2], ignore_index=True, sort=False)


def split_x_y(df):
    x = list(df['text'])
    y = df[list(set(df.columns) - {'text'})]
    return x, y


def process_dataset(path, emotions):
    dialogues = split_on_dialogue(path)
    labelled = []
    for d in dialogues:
        text = clean_text(d)
        label = create_label(text, emotions)
        if (text, label) not in labelled:
            labelled.append((text, label))
    df = create_df(labelled)
    return df


def individual_labels(y, unique_emotions):
    return [y[[emotion]].values for emotion in unique_emotions]


def embedded_vectors(GloVe_path, X_train, X_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, padding='post', maxlen=200)
    X_test = pad_sequences(X_test, padding='post', maxlen=200)

    v_size = len(tokenizer.word_index) + 1

    embed_dict = dict()
    _, _, files = next(walk(GloVe_path))

    for file in files:
        with open(GloVe_path+file, 'r') as file:
            for line in file:
                records = line.split()
                embed_dict[records[0]] = np.asarray(records[1:], dtype='float32')

    matrix = np.zeros((v_size, 100))
    for word, index in tokenizer.word_index.items():
        vector = embed_dict.get(word)
        if vector is not None:
            matrix[index] = vector

    return X_train, X_test, v_size, matrix, tokenizer


def preprocessing(paths):
    print('Starting to preprocess...')
    # paths to content of the data folder
    emotions_path = paths[0]
    GloVe_path = paths[1]
    COVID_file = paths[2]
    MED_file = paths[3]

    # path to content of the stored folder
    saved_path = paths[4]

    unique_emotions, emotions = create_emotions(emotions_path)

    print('Preprocessing COVID-Dialogue-Dataset-English...')
    COVID_df = process_dataset(COVID_file, emotions)

    print('Preprocessing MedDialog dataset (English)...')
    MED_df = process_dataset(MED_file, emotions)

    merged_df = binarizer(append_dfs(COVID_df, MED_df))
    print('Preprocessing done!')

    return COVID_df, MED_df, unique_emotions, merged_df


def converting(merged_df, unique_emotions, paths):
    print('Starting to convert...')
    GloVe_path = paths[1]

    X, y = split_x_y(merged_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True)

    y_train = individual_labels(y_train, unique_emotions)
    y_test = individual_labels(y_test, unique_emotions)

    X_train, X_test, v_size, matrix, tokenizer = embedded_vectors(GloVe_path, X_train, X_test)
    print('Converting done!')

    return X_train, X_test, y_train, y_test, v_size, matrix, tokenizer


def convert_df_to_num(tokenizer, COVID_df, MED_df):
    COVID_list = COVID_df['text'].tolist()
    MED_list = MED_df['text'].tolist()

    COVID = tokenizer.texts_to_sequences(COVID_list)
    MED = tokenizer.texts_to_sequences(MED_list)

    COVID_vec = pad_sequences(COVID, padding='post', maxlen=200)
    MED_vec = pad_sequences(MED, padding='post', maxlen=200)
    return COVID_vec, MED_vec


#######################
# CLASSIFICATION MODEL
#######################


def create_model(v_size, matrix):
    input_1 = Input(shape=(200,))
    embedding_layer = Embedding(v_size, 100, weights=[matrix],
                                trainable=False)(input_1)
    LSTM_Layer1 = LSTM(128)(embedding_layer)

    output1 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output2 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output3 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output4 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output5 = Dense(1, activation='sigmoid')(LSTM_Layer1)

    model = Model(inputs=input_1, outputs=[output1, output2, output3, output4,
                                           output5])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['acc'])

    return model


def main_model(X_train, y_train, v_size, matrix, epochs=10):
    print('Creating the model...')
    model = create_model(v_size, matrix)
    print('Training the model...')
    history = model.fit(x=X_train, y=y_train, batch_size=8192, epochs=epochs,
                        verbose=1, validation_split=0.2)
    print('Done!')
    return model, history


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(x=X_test, y=y_test, verbose=1, return_dict=True)
    loss = ['dense_loss', 'dense_1_loss', 'dense_2_loss', 'dense_3_loss',
              'dense_4_loss']
    acc = ['dense_acc', 'dense_1_acc', 'dense_2_acc', 'dense_3_acc',
            'dense_4_acc']
    layers = ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']

    print('\nTotal loss: ', score['loss'])
    for i in range(len(layers)):
        print('')
        print(layers[i], 'loss: ', score[loss[i]])
        print(layers[i], 'accuracy: ', score[acc[i]])


#############
# EVALUATION
#############


def plot_loss(history):
    df = pd.DataFrame(columns=['method', 'Epoch #', 'Loss'])

    for i in range(len(history['loss'])):
        df = df.append({'method': 'loss', 'Epoch #': i,
                        'Loss': history['loss'][i]}, ignore_index=True)
    for i in range(len(history['val_loss'])):
        df = df.append({'method': 'val_loss', 'Epoch #': i,
                        'Loss': history['val_loss'][i]}, ignore_index=True)

    sns.lineplot(x='Epoch #', y='Loss', data=df, hue='method').set_title('Total Loss')
    plt.show()

    denses = ['dense_loss', 'dense_1_loss', 'dense_2_loss', 'dense_3_loss',
              'dense_4_loss']
    val_denses = ['val_dense_loss', 'val_dense_1_loss', 'val_dense_2_loss',
                  'val_dense_3_loss', 'val_dense_4_loss']
    titles = ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']

    for d in range(len(denses)):
        df = pd.DataFrame(columns=['method', 'Epoch #', 'Loss'])

        hd = history[denses[d]]
        for i in range(len(hd)):
            df = df.append({'method': 'loss', 'Epoch #': i, 'Loss': hd[i]},
                           ignore_index=True)
        hvd = history[val_denses[d]]
        for i in range(len(hvd)):
            df = df.append({'method': 'val_loss', 'Epoch #': i, 'Loss': hvd[i]},
                           ignore_index=True)

        sns.lineplot(x='Epoch #', y='Loss', data=df, hue='method').set_title('Loss for '+ titles[d])
        plt.show()


def plot_acc(history):
    denses = ['dense_acc', 'dense_1_acc', 'dense_2_acc', 'dense_3_acc',
              'dense_4_acc']
    val_denses = ['val_dense_acc', 'val_dense_1_acc', 'val_dense_2_acc', 'val_dense_3_acc',
                  'val_dense_4_acc']
    titles = ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']

    for d in range(len(denses)):
        df = pd.DataFrame(columns=['method', 'Epoch #', 'Loss'])

        hd = history[denses[d]]
        for i in range(len(hd)):
            df = df.append({'method': 'acc', 'Epoch #': i, 'Accuracy': hd[i]},
                           ignore_index=True)
        hvd = history[val_denses[d]]
        for i in range(len(hvd)):
            df = df.append({'method': 'val_acc', 'Epoch #': i, 'Accuracy': hvd[i]},
                           ignore_index=True)

        sns.lineplot(x='Epoch #', y='Accuracy', data=df, hue='method').set_title('Accuracy for '+ titles[d])
        plt.show()


#############
# ANNOTATION
#############


def annotate(df, unique_emotions):
    annotated = defaultdict(int)
    for row in df['labels']:
        for emotion in unique_emotions:
            if emotion in set(row):
                annotated[emotion] += 1
    return OrderedDict(sorted(annotated.items()))


def annotate_with_model(X, model, unique_emotions):
    y = model.predict(X)
    annotated = dict()
    for i in range(len(y)):
        annotated[unique_emotions[i]] = len([p for p in y[i] if p >= 0.5])
    return OrderedDict(sorted(annotated.items()))


def compare_annotation(a, a_model, title):

    df = pd.DataFrame(columns=['method', 'emotion', 'count'])

    for emotion, count in a.items():
        df = df.append({'method': 'lexicon', 'emotion': emotion,
                        'count': count}, ignore_index=True)
    for emotion, count in a_model.items():
        df = df.append({'method': 'trained model', 'emotion': emotion,
                        'count': count}, ignore_index=True)

    sns.barplot(x='emotion', y='count', data=df, hue='method').set_title(title)
    plt.show()


###########
# RESULTS
###########


def normalize_dict(d):
    factor = 100/sum(d.values())
    return {key: float(value*factor) for key, value in d.items()}


def create_plot_df(MED, COVID):
    df = pd.DataFrame(columns=['dataset', 'emotion', 'count'])
    for emotion, percent in MED.items():
        df = df.append({'dataset': 'before', 'emotion': emotion,
                        'percentage': percent}, ignore_index=True)
    for emotion, percent in COVID.items():
        df = df.append({'dataset': 'after', 'emotion': emotion,
                        'percentage': percent}, ignore_index=True)
    return df


def compare_emotions(MED, COVID, title):
    MED = OrderedDict(sorted(normalize_dict(MED).items()))
    COVID = OrderedDict(sorted(normalize_dict(COVID).items()))
    df = create_plot_df(MED, COVID)
    sns.barplot(x='emotion', y='percentage', data=df, hue='dataset').set_title(title)
    plt.show()
