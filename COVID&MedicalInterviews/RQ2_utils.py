#############################
# THIS FILE IS MADE BY:
# SHELBY JHORAI (ID:11226374)
#
# CONTENT:
# - Load
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


#######
# LOAD
#######


def load_variables(saved_path):
    """
    Input: file path.
    Returns list of unique emotions, vocabulary size and matrix.
    """
    with open(saved_path+'unique_emotions.txt', 'rb') as filehandle:
        unique_emotions = pickle.load(filehandle)
    with open(saved_path+'v_size.txt', 'rb') as filehandle:
        v_size = pickle.load(filehandle)
    with open(saved_path+'matrix.txt', 'rb') as filehandle:
        matrix = pickle.load(filehandle)
    return unique_emotions, v_size, matrix


def load_x_y(saved_path):
    """
    Input: file path.
    Returns train and test set.
    """
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
    """
    Input: file path.
    Returns numerical representation of both datasets.
    """
    with open(saved_path+'C_vec.txt', 'rb') as filehandle:
        C_vec = pickle.load(filehandle)
    with open(saved_path+'M_vec.txt', 'rb') as filehandle:
        M_vec = pickle.load(filehandle)

    return C_vec, M_vec


def load_dfs(saved_path):
    """
    Input: file path.
    Returns dataframe of both datasets.
    """
    with open(saved_path+'COVID_df.txt', 'rb') as filehandle:
        COVID_df = pickle.load(filehandle)
    with open(saved_path+'MED_df.txt', 'rb') as filehandle:
        MED_df = pickle.load(filehandle)
    return COVID_df, MED_df


def load_trained_model(path, name):
    """
    Input: file path and file name.
    Returns trained multi-label classification model.
    """
    return load_model(path+name)


def load_history(saved_path, name):
    """
    Input: file path and file name.
    Returns history of the model.
    """
    with open(saved_path+name, 'rb') as filehandle:
        history = pickle.load(filehandle)
    return history


################
# PREPROCESSING
################


def split_on_dialogue(file_path):
    """
    Input: file path.
    Returns list with dialogues.
    """
    with open(file_path, encoding='utf8') as f:
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
    """
    Input: file path.
    Returns list of unique emotions and dictionary with as key a word
    and as value the corresponding emotion.
    """
    emotions = dict()
    unique_emotions = []
    lemmatizer = WordNetLemmatizer()
    _, _, files = next(walk(emotions_path))
    
    # Process each emotion-score file in the folder.
    for file in files:
        emotion = file.replace('-scores.txt', '')

        # Add emotion to the list of unique emotions.
        unique_emotions.append(emotion)

        with open(emotions_path+file, 'r', encoding='utf8') as f:
            for line in f:
                word, p = line.split('\t')

                # Add word to dictionary if the intensity score is greater than 0.6.
                if float(p) > 0.6:
                    word = lemmatizer.lemmatize(word)
                    emotions[word] = emotion
    return unique_emotions, emotions


def remove_noise(token):
    """
    Input: token.
    Returns token without URLs, punctuation and next line symbols.
    """
    # Remove next line symbols.
    token = re.sub(r"\\n", "", token)
    token = re.sub(r"\n", "", token)

    # Remove URLs.
    token = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                   r'(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)

    # Remove punctuation and turn token to lower-case.
    token = re.sub(r"(@[A-Za-z0-9_]+)", "", token)
    token = re.sub(r"-", " ", token)
    token = token.lower().translate(str.maketrans('', '', string.punctuation))
    return token


def lemmatize(token, tag):
    """
    Input: token and POS-tag.
    Returns lemmatized token.
    """
    lemmatizer = WordNetLemmatizer()
    pos = False

    # Check if token is a noun.
    if tag.startswith("NN"):
        pos = 'n'

    # Check if token is a verb.
    elif tag.startswith('VB'):
        pos = 'v'

    # Check if token is an adjective.
    elif tag.startswith('JJ'):
        pos = 'a'
    if pos:
        return lemmatizer.lemmatize(token, pos)

    # If token is not a noun, verb or adjective, return empty string.
    else:
        return ''


def clean_text(raw_text):
    """
    Input: text sequence
    Returns list of preprocessed tokens.
    """
    cleaned = []

    # Define set of stopwords.
    stop_words = set(stopwords.words('english')) | set(['http', 'patient', 'doctor'])
    
    # Tokenize text sequence.
    tokens = word_tokenize(str(raw_text), "english")

    # Give tokens a POS tag.
    tagged_tokens = pos_tag(tokens)
    for token, tag in tagged_tokens:

        # Remove noise and stopwords, and lemmatize token.
        token = remove_noise(token)
        token = lemmatize(token, tag)
        if len(token) > 2 and len(token) < 20 and token not in stop_words:
            cleaned.append(token)

    return cleaned


def annotate_with_lexicon(text, emotions):
    """
    Input: text = list of tokens, emotions = dictionary with as key a word
    and as value the corresponding emotion.
    Returns list of unique emotions associated with the text.
    """
    return set([emotions[token] for token in text if token in emotions.keys()])


def create_df(labelled):
    """
    Input: list of tuples containing text and label.
    Returns Dataframe of list with as columns 'text' and 'labels'.
    """
    df = pd.DataFrame(columns=['text', 'labels'])
    for text, label in labelled:
        df = df.append({'text': text, 'labels': label}, ignore_index=True)
    return df


def append_dfs(df1, df2):
    """
    Input: df1 = Dataframe, df2 = Dataframe.
    Returns one Dataframe with df1 and df2.
    """
    # Get two Dataframes of equal length.
    if len(df1) < len(df2):
        df2 = df2.sample(len(df1))
    elif len(df2) < len(df1):
        df1 = df1.sample(len(df2))

    return pd.concat([df1, df2], ignore_index=True, sort=False)


def split_x_y(df):
    """
    Input: Dataframe containing 'text' and label columns.
    Returns X and y set of the Dataframe.
    """
    # Get column 'text' from Dataframe.
    x = list(df['text'])
    
    # Get label columns from Dataframe.
    y = df[list(set(df.columns) - {'text'})]

    return x, y


def process_dataset(path, emotions):
    """
    Input: path = file path, emotions = dictionary with as key a word
    and as value the corresponding emotion.
    Returns Dataframe with preprocessed text and binarized labels.
    """
    # Get dialogues from file.
    dialogues = split_on_dialogue(path)
    labelled = []
    
    # Process each dialogue.
    for d in dialogues:
        
        # Remove noise and lemmatize tokens
        text = clean_text(d)
        
        # Annotate dialogues with lexicon
        label = annotate_with_lexicon(text, emotions)
        
        # Add tuple of text and label to list if it is not in list yet.
        if (text, label) not in labelled:
            labelled.append((text, label))

    # Create Dataframe of list with tuples.
    df = create_df(labelled)

    return df


def binarizer(df):
    """
    Input: Dataframe containing 'text' and 'labels'
    Returns Dataframe with binarized labels.
    """
    # Binarize labels.
    mlb = MultiLabelBinarizer()
    result = mlb.fit_transform(df['labels'])
    
    # Create Dataframe with as columns 'text' and each emotion.
    new_df = pd.concat([df['text'],
                        pd.DataFrame(result, columns=list(mlb.classes_))],
                       axis=1)
    return new_df


def preprocessing(paths):
    """
    Input: list of file paths.
    Returns preprocessed dataframes of both datasets, a list of unique emotions
    and a merged dataframe of both dataframes.
    """
    print('Starting to preprocess...')
    # Paths to content of the data folder.
    emotions_path = paths[0]
    GloVe_path = paths[1]
    COVID_file = paths[2]
    MED_file = paths[3]

    # Path to content of the stored folder.
    saved_path = paths[4]

    unique_emotions, emotions = create_emotions(emotions_path)

    print('Preprocessing COVID-Dialogue-Dataset-English...')
    COVID_df = process_dataset(COVID_file, emotions)

    print('Preprocessing MedDialog dataset (English)...')
    MED_df = process_dataset(MED_file, emotions)

    merged_df = binarizer(append_dfs(COVID_df, MED_df))
    print('Preprocessing done!')

    return COVID_df, MED_df, unique_emotions, merged_df


def individual_labels(y, unique_emotions):
    """
    Input: y = Dataframe with as columns the unique emotions,
    unique_emotions = list of unique emotions
    Returns list containing lists with values for each unique emotion.
    """
    return [y[[emotion]].values for emotion in unique_emotions]


def embedded_vectors(GloVe_path, X_train, X_test):
    """
    Input: GloVe_path = file path, X_train = text of training set,
    X_test = text of test set.
    Returns numerical representation of X_train and X_test, vocabulary size, matrix
    and tokenizer
    """
    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    # Fit tokenizer on train set.
    tokenizer.fit_on_texts(X_train)

    # Convert training and test set.
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad training and test set to length of 200.
    X_train = pad_sequences(X_train, padding='post', maxlen=200)
    X_test = pad_sequences(X_test, padding='post', maxlen=200)

    # Get size of vocabulary.
    v_size = len(tokenizer.word_index) + 1

    embed_dict = dict()
    
    # Get files from GloVe folder.
    _, _, files = next(walk(GloVe_path))

    # Read each GloVe file into a dictionary.
    for file in files:
        with open(GloVe_path+file, 'r', encoding='utf8') as file:
            for line in file:
                records = line.split()
                embed_dict[records[0]] = np.asarray(records[1:], dtype='float32')

    # Create weight matrix.
    matrix = np.zeros((v_size, 100))
    for word, index in tokenizer.word_index.items():
        vector = embed_dict.get(word)
        if vector is not None:
            matrix[index] = vector

    return X_train, X_test, v_size, matrix, tokenizer


def converting(merged_df, unique_emotions, paths):
    """
    Input: merged_df = Dataframe containing both preprocessed datasets,
    unique_emotions = list of unique emotions, paths = list of file paths.
    Returns training and test set, vocabulary size, weight matix and tokenizer.
    """
    print('Starting to convert...')
    GloVe_path = paths[1]

    # Separate text and labels from Dataframe.
    X, y = split_x_y(merged_df)

    # Create training and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True)

    # Get individual label for each emotion.
    y_train = individual_labels(y_train, unique_emotions)
    y_test = individual_labels(y_test, unique_emotions)

    # Create numerical representation.
    X_train, X_test, v_size, matrix, tokenizer = embedded_vectors(GloVe_path, X_train, X_test)
    print('Converting done!')

    return X_train, X_test, y_train, y_test, v_size, matrix, tokenizer


def convert_df_to_num(tokenizer, COVID_df, MED_df):
    """
    Input: tokenizer = tokenizer fitted on training set, COVID_df = Dataframe
    containing COVID dataset, MED_df = Dataframe containing MedDialog dataset.
    Returns numerical representation of text in both dataframes.
    """
    # Turn text columns of dataframes into lists.
    COVID_list = COVID_df['text'].tolist()
    MED_list = MED_df['text'].tolist()

    # Convert lists.
    COVID = tokenizer.texts_to_sequences(COVID_list)
    MED = tokenizer.texts_to_sequences(MED_list)

    # Pad to same length.
    COVID_vec = pad_sequences(COVID, padding='post', maxlen=200)
    MED_vec = pad_sequences(MED, padding='post', maxlen=200)
    return COVID_vec, MED_vec


#######################
# CLASSIFICATION MODEL
#######################


def create_model(v_size, matrix):
    """
    Input: vocabulary size and weight matrix.
    Returns multi-label classification model.
    """
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


def main_model(X_train, y_train, v_size, matrix, epochs):
    """
    Input: training set, vocabulary size, weights matrix and number of epochs.
    This functions creates and trains a multi-label classification model.
    Returns the model and the training history.
    """
    print('Creating the model...')
    model = create_model(v_size, matrix)
    print('Training the model...')
    history = model.fit(x=X_train, y=y_train, batch_size=8192, epochs=epochs,
                        verbose=1, validation_split=0.2)
    print('Done!')
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Input: trained multi-label classification model and test set.
    Prints loss and accuracy of each dense output layer.
    """
    # Evaluate the model on test set
    score = model.evaluate(x=X_test, y=y_test, verbose=1, return_dict=True)
    
    # Print loss and accuracy for each dense output layer.
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
    """
    Input: training history of the model.
    Plots total loss and losses of each dense output layer.
    """
    # Initialize Dataframe to plot.
    df = pd.DataFrame(columns=['method', 'Epoch #', 'Loss'])

    # Add total loss values to dataframe.
    for i in range(len(history['loss'])):
        df = df.append({'method': 'loss', 'Epoch #': i,
                        'Loss': history['loss'][i]}, ignore_index=True)
    for i in range(len(history['val_loss'])):
        df = df.append({'method': 'val_loss', 'Epoch #': i,
                        'Loss': history['val_loss'][i]}, ignore_index=True)

    # Plot total loss.
    sns.lineplot(x='Epoch #', y='Loss', data=df, hue='method').set_title('Total Loss')
    plt.show()

    denses = ['dense_loss', 'dense_1_loss', 'dense_2_loss', 'dense_3_loss',
              'dense_4_loss']
    val_denses = ['val_dense_loss', 'val_dense_1_loss', 'val_dense_2_loss',
                  'val_dense_3_loss', 'val_dense_4_loss']
    titles = ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']

    # Plot losses for each dense output layer.
    for d in range(len(denses)):

        # Initialize Dataframe to plot.
        df = pd.DataFrame(columns=['method', 'Epoch #', 'Loss'])

        # Add loss values to dataframe.
        hd = history[denses[d]]
        for i in range(len(hd)):
            df = df.append({'method': 'loss', 'Epoch #': i, 'Loss': hd[i]},
                           ignore_index=True)
        hvd = history[val_denses[d]]
        for i in range(len(hvd)):
            df = df.append({'method': 'val_loss', 'Epoch #': i, 'Loss': hvd[i]},
                           ignore_index=True)

        # Plot loss of the current dense output layer.
        sns.lineplot(x='Epoch #', y='Loss', data=df, hue='method').set_title('Loss for '+ titles[d])
        plt.show()


def plot_acc(history):
    """
    Input: training history of the model.
    Plots accuracy of each dense output layer.
    """
    denses = ['dense_acc', 'dense_1_acc', 'dense_2_acc', 'dense_3_acc',
              'dense_4_acc']
    val_denses = ['val_dense_acc', 'val_dense_1_acc', 'val_dense_2_acc', 'val_dense_3_acc',
                  'val_dense_4_acc']
    titles = ['dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']

    # Plot accuracy for each dense output layer.
    for d in range(len(denses)):

        # Initialize Dataframe to plot.
        df = pd.DataFrame(columns=['method', 'Epoch #', 'Loss'])

        # Add loss values to dataframe.
        hd = history[denses[d]]
        for i in range(len(hd)):
            df = df.append({'method': 'acc', 'Epoch #': i, 'Accuracy': hd[i]},
                           ignore_index=True)
        hvd = history[val_denses[d]]
        for i in range(len(hvd)):
            df = df.append({'method': 'val_acc', 'Epoch #': i, 'Accuracy': hvd[i]},
                           ignore_index=True)

        # Plot accuracy of the current dense output layer.
        sns.lineplot(x='Epoch #', y='Accuracy', data=df, hue='method').set_title('Accuracy for '+ titles[d])
        plt.show()


#############
# ANNOTATION
#############


def convert_df_to_dict(df):
    """
    Input: df = Dataframe with column 'labels'
    Returns dictionary with as key the emotion and as value the amount
    of occurences in the dataframe.
    """
    annotated = defaultdict(int)

    # Process each label in the Dataframe.
    for row in df['labels']:
        
        # For each emotion, if the emotion occurs in the label
        # increase the count.
        
        for emotion in row:
            annotated[emotion] += 1

    # Return sorted dictionary.
    return OrderedDict(sorted(annotated.items()))


def annotate_with_model(X, model, unique_emotions):
    """
    Input: X = numerical representation of the dialogues,
    model = trained muti-label classification model,
    unique_emotions = list of unique emotions
    Returns dictionary with as key the emotion and as value the amount
    of occurences in the dataframe.
    """
    # Predict labels with trained model.
    y = model.predict(X)
    annotated = dict()
    
    # For each emotion, add emotions to dictionary with as value the amount
    # of occurences in the dataframe.
    for i in range(len(y)):
        
        # Only add emotion if prediction is equal or greater than 0.5.
        annotated[unique_emotions[i]] = len([p for p in y[i] if p >= 0.5])

    # Return sorted dictionary.
    return OrderedDict(sorted(annotated.items()))


def plot_annotation(a, title):
    """
    Input: a = dictionary with as key the emotion and as value the count,
    title = title for the figure.
    Plots annotation of a dataset.
    """
    # Transform dictionary to DataFrame
    df = pd.DataFrame(a.items())

    # Plot Dataframe.
    xl = 'emotions'
    yl = 'count'
    sns.barplot(x=0, y=1, data=df,color='tab:blue').set(xlabel=xl, ylabel=yl, title=title)
    plt.show()


def compare_annotation(a, a_model, title):
    """
    Input: a = annotion with lexicon, a_model = annotation with trained model,
    title = title for the figure.
    Plots annotation differences of a dataset.
    """
    # Initialize Dataframe to plot.
    df = pd.DataFrame(columns=['method', 'emotion', 'count'])

    # Add annotations to the dataframe.
    for emotion, count in a.items():
        df = df.append({'method': 'lexicon', 'emotion': emotion,
                        'count': count}, ignore_index=True)
    for emotion, count in a_model.items():
        df = df.append({'method': 'trained model', 'emotion': emotion,
                        'count': count}, ignore_index=True)

    # Plot Dataframe.
    sns.barplot(x='emotion', y='count', data=df, hue='method').set_title(title)
    plt.show()


###########
# RESULTS
###########


def normalize_dict(d):
    """
    Input: dictionary.
    Returns normalized dictionary.
    """
    factor = 100/sum(d.values())
    return {key: float(value*factor) for key, value in d.items()}


def create_plot_df(MED, COVID):
    """
    Input: MED = dictionary with as key the emotion and as value the percentage
    of occurences in the dataframe for the MedDialog dataset,
    COVID = dictionary with as key the emotion and as value the percentage
    of occurences in the dataframe for the COVID dataset.
    Returns Dataframe to plot.
    """
    # Initialize Dataframe.
    df = pd.DataFrame(columns=['dataset', 'emotion', 'count'])

    # Add dictionaries to the Dataframe.
    for emotion, percent in MED.items():
        df = df.append({'dataset': 'before', 'emotion': emotion,
                        'percentage': percent}, ignore_index=True)
    for emotion, percent in COVID.items():
        df = df.append({'dataset': 'after', 'emotion': emotion,
                        'percentage': percent}, ignore_index=True)
    return df


def compare_emotions(MED, COVID, title):
    """
    Input: ED = dictionary with as key the emotion and as value the amount
    of occurences in the dataframe for the MedDialog dataset,
    COVID = dictionary with as key the emotion and as value the amount
    of occurences in the dataframe for the COVID dataset,
    title = title for the figure.
    Plots emotion distribution.
    """
    # Normalize dictionaries.
    MED = OrderedDict(sorted(normalize_dict(MED).items()))
    COVID = OrderedDict(sorted(normalize_dict(COVID).items()))

    # Create Dataframe to plot.
    df = create_plot_df(MED, COVID)

    # Plot emotion distribution.
    sns.barplot(x='emotion', y='percentage', data=df, hue='dataset').set_title(title)
    plt.show()
