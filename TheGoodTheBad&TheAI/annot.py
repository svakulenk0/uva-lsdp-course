import nltk
import pandas as pd
import convokit
from convokit import Corpus, download

all_csvs = ["Datasets/together_StarWars_sinmpel.csv", "Datasets/togetherAustinP_simpel.csv", 
            "Datasets/togetherbatman_simpel.csv", "Datasets/togetherjurassicpark_simpel.csv",
            "Datasets/togetherlambs_simpel.csv", "Datasets/togethermatrix_simpel.csv",
            "Datasets/togethermib_simpel.csv", "Datasets/togetherNightmareXmass_simpel.csv",
            "Datasets/togetherPoC_simpel.csv"]



df_convomet = pd.DataFrame({'Dataset':['Convokit'], '# of utterances':[3820], '# of dialogues':[1154], '# of agents':[115], 'average worsd/utt':[60.48]})

df_cmumet = pd.DataFrame({'Dataset':['CMU'], '# of movies':[39568], '# of genres':[24], 'average words/plot':[1771.6]})

df_cetmet = pd.DataFrame({'Dataset':['cetinsamet'], '# of movies':[7868], '# of genres':[8], 'average line/movie':[1177] , 'average line length':[12.62] })

df_BPSmet = pd.DataFrame({'Dataset':['BPS'], '# of utterances': [1308] , '# of agents':[5], '# of words/utt':[159.3]  })


def annot(corpus, movie_title):
    conversation = list()
    df = pd.DataFrame()
    # Make list of all conversations
    utterances = list(corpus.iter_utterances())
    # For each conversation in movie
    for utt in utterances:
        if utt.speaker.meta['movie_name'] == movie_title:
            # If first loop
            if len(conversation) == 0:
                conversation.append(utt)
                conversation_id = utt.conversation_id
            # Next sentence in convo
            elif utt.conversation_id == conversation_id:
                conversation.append(utt)
            # If new conversation
            else:
                #annotations = parse(conversation)
                df = annotate(corpus, df, conversation, conversation_id, {})
                conversation = [utt]
                conversation_id = utt.conversation_id
    #annotations = parse(conversation)
    df = annotate(corpus, df, conversation, conversation_id, {})
    df = add_char_lab(df)
    return df

# Print utterances + speaker + etc
def parse(convo):
    annot = dict()
    # Print ze in de juiste volgorde
    for utt in convo[::-1]:
        print(utt.speaker.meta['character_name']+':')
        print(utt.text+'\n')
        print('--------------------------------------\n')
        annot[utt.speaker.meta['character_name']] = 0
    # Create dict of chars in convo
    for char in annot:
        print(char)
        annot[char] = input()
    print('-------------------------------------------------------\n')
    print('-------------------------------------------------------\n')
    return annot

# Saves the annotations with relevant data in df
def annotate(corpus, df, conversation, conversation_id, annotations):
    for utt in conversation:
        conv = corpus.get_conversation(conversation_id)
        if len(annotations) > 0 :
            row = [utt.speaker.meta['movie_name'], conv.meta['genre'], utt.speaker.meta['character_name'], 
                   conversation_id, utt.text, annotations[utt.speaker.meta['character_name']]]
        else:
            row = [utt.speaker.meta['movie_name'], conv.meta['genre'], utt.speaker.meta['character_name'], 
                   conversation_id, utt.text, 3]
        df_row = pd.DataFrame([row])
        df_row = df_row.rename(columns={0:'Movie name', 1:'Movie genre', 2:'Speaker',
                               3:'Conversation id', 4:'utterance', 5:'Intent label'})
        
        df = pd.concat([df, df_row],ignore_index=True)
    return df

# Add a generic char label for each speaker
def add_char_lab(df):
    speakers = df['Speaker'].unique()
    print('-----------------------------')
    print("CHARACTER LABELING STARTS NOW")
    df["Character_label"] = [0 for i in range(len(df))]
    for speaker in speakers:
        print(speaker +'\n')
        char_lab = input()
        df.loc[(df['Speaker'] == speaker), 'Character_label'] = char_lab
    return df

def save_annots(df, title):
    # Save to csv
    df.to_csv(title+'.csv', index=False)
    print('Saved as: '+title+'.csv')
    return 1
