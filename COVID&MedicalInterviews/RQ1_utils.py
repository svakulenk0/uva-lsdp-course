# Imports
import nltk
from nltk.tokenize import word_tokenize
from nltk import tokenize
nltk.download('punkt')
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import os

# Data Preprocessing
def split_on_dialogue(data_path):
    """
    Returns list with conversations
    Format conversatoins: [[conversation1], [conversation2], ..., [conversation_n]]
    """

    with open(data_path, encoding='utf8') as f:
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
            dialogue_i = i + 1

        if i == dialogue_i + j:
            convo.append(line)
            j += 1
            if len(tokens) == 0:
                conversations.append(convo)
                convo = []
                j = 0
                continue
    return conversations


def split_on_sentences(conversations):
    """
    A Function that splits the conversations in sentences.
    """

    sentence_list = []

    for conversation in conversations:
        for sentences in conversation:
            token_sen = tokenize.sent_tokenize(sentences)
            for sentence in token_sen:
                if sentence != 'Patient:' and sentence != 'Doctor:':
                    sentence_list.append(sentence)

    return sentence_list


def save(df, save_preprocessed_dataframe_path, name):
    """
    Function that saves the created dataframe as a csv.
    """

    df.to_csv(save_preprocessed_dataframe_path + name + '.csv', index=False)


def preprocess_to_csv(data_path, save_to):
    """
    A function that preprocesses the data (so that it is displayed per sentence),
    and saves is as a .csv file for later use.
    """

    # Split on dialogue
    conversations = split_on_dialogue(data_path)

    # Split on sentence
    sentences = split_on_sentences(conversations)

    # Make dataframe and drop dubplicates
    df_sent = pd.DataFrame(np.array(sentences), columns=['sentences'])
    df_sent.drop_duplicates(keep='first', inplace=True)

    # Save
    name = os.path.basename(data_path)
    save(df_sent, save_to, name[:-4])

    print(name, 'done')



# Evaluation
def get_predicted_symptoms(prediction):
    """
    This function takes in the prediction of a sentence of the pre-trained model 
    and returns the symptoms mentioned in that sentence.
    """
    
    symptoms = []
    
    # Check if there is a predicted entity
    if len(prediction[0]['entity']) > 0:
        number_of_entities = len(prediction[0]['entity'])
    
        # Loop over predicted entities and get symptoms (here called: disease)
        for i in range(number_of_entities):
            if prediction[0]['entity'][i]['type'] == 'disease':
                symptoms.append(prediction[0]['entity'][i]['mention'])
            
    return symptoms 

    
def eval(df, model):
    """
    This function computes the accuracy score, given a dataframe.
    """
    number_of_symptoms = 0
    TP = 0
    FP = 0

    for ind in tqdm(df.index):

        sentence = df['sentences'][ind]

        # Padding is needed because algorithms is not used to small sentences
        if len(sentence) < 60:
            sentence = sentence + '...'

        prediction = model.predict([sentence])

        predicted_symptom = get_predicted_symptoms(prediction)
        predicted_symptom = [x.lower() for x in predicted_symptom]
        predicted_symptom = [x.split(', ')[0] for x in predicted_symptom]
        gt_symptom = df['symptoms'][ind]

        # If it's not nan
        if isinstance(gt_symptom, str):
            gt_list = gt_symptom.split(', ')

            # Keep track of symptoms
            for symptom in gt_list:
                number_of_symptoms += 1

                # Keep track of well predicted symptoms
                if symptom in predicted_symptom:
                    TP += 1
                else:
                    FP += 1

    accuracy = TP/number_of_symptoms
    
    return accuracy, [TP, FP],  number_of_symptoms
