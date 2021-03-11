import json
import pandas as pd
import nltk
from collections import Counter
import operator
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


# get list of all paths to the json-files of english episodes given subset number (bart: 0 , juno: 1, joris: 2)
def get_paths_for_en_episodes(subset_number):
    """
    Function returns list of all paths to the json-files of english 
    episodes given subset number (bart: 0 , juno: 1, joris: 2)
    
    """
    metadata_df = pd.read_csv("podcast_data_no_audio/metadata/metadata.tsv",sep='\t')
    path1 = 'podcast_data_no_audio/podcasts-transcripts/' + str(subset_number)

    folders = listdir(path1)

    if '.DS_Store' in folders:
        folders.remove('.DS_Store')

    podcast_episodes_paths = []

    for letter_or_number in tqdm(folders):    
        path2 = path1 + '/' + letter_or_number


        for show_uri in listdir(path2):
            path3 = path2 + '/' + show_uri

            # select english shows only
            show_metadata = metadata_df.loc[metadata_df['show_filename_prefix'] == show_uri]

            if len(show_metadata['language'].unique()) > 0:
                if 'en' in show_metadata['language'].unique()[0]:
                    for episode_uri in listdir(path3):
                        path4 = path3 + '/' + episode_uri

                        if '.json' in path4:
                            podcast_episodes_paths.append(path4)

                
        
    return len(podcast_episodes_paths), podcast_episodes_paths

## Belangrijke functie voor representatie van een podcast episode ##

def dialogue_json_to_pandas(json_path):
    """
    This function converts a podcast .json transcript into a 
    pandas dataframe with speaker tags, utterance text and open labels
    
    """
    
    with open(json_path) as f:
        data = json.load(f)

    # get transcript parts from json file, remove empty parts
    transcript_parts = []
    for utt in data['results']:
        try:
            trans = utt['alternatives'][0]['transcript']
        except KeyError:
            trans = 0

        if trans != 0:
            transcript_parts.append(utt)
    

    # create list of sentences from dialogue
    sentences = []
    for index, utterance in enumerate(transcript_parts):

        # get text of utterance
        utterance_text = utterance['alternatives'][0]['transcript']
        
        # get sentences from text to split based on speakerTag
        utterance_sentences = nltk.sent_tokenize(utterance_text)
        for sent in utterance_sentences:
            sent = sent.split(" ")
            if '' in sent:
                sent.remove('')
            sentences.append(sent)
                
    
    # get words with tags from transcript file
    words_with_tags = data['results'][-1]['alternatives'][0]['words']
    
    
    # assign speakerTag to each sentence
    # also fix mistakes when speakerTag switches to other speaker
    # in the middle of a sentence
    sentences_with_tags = []
    
    word_idx = 0
    for index, sentence in enumerate(sentences):
        sent_with_tags = []
        for word in sentence:
            sent_with_tags.append((word, words_with_tags[word_idx]['speakerTag']))
            word_idx += 1
        
        c = Counter(elem[1] for elem in sent_with_tags)
        
        sent_speakerTag = max(c.items(), key=operator.itemgetter(1))[0]
        
        
        sentences_with_tags.append((' '.join(sentence), sent_speakerTag))
        
        
    # merge sentences with same consecutive tags
    utterances_texts = []
    utterances_tags = []
    merged_sents = []
    for index, tagged_sent in enumerate(sentences_with_tags):

        
        # set initial value for tagged_sent
        if index == 0:
            curr_tag = tagged_sent[1]
        
        # speaker switch
        if curr_tag != tagged_sent[1] and index > 0:
             
            utterance_tag = merged_sents[0][1]
            utterance_text = ' '.join([sent[0] for sent in merged_sents])

            utterances_texts.append(utterance_text)
            utterances_tags.append(utterance_tag)
            merged_sents = []

        curr_tag = tagged_sent[1]
        merged_sents.append(tagged_sent)
        
        if index == len(sentences_with_tags)-1:
            utterance_tag = merged_sents[0][1]
            utterance_text = ' '.join([sent[0] for sent in merged_sents])

            utterances_texts.append(utterance_text)
            utterances_tags.append(utterance_tag)
            
            
   
    # make utterances and tags are the same shape
    if len(utterances_texts) == len(utterances_tags):
        
        # create pandas dataframe
        dialogue_df = pd.DataFrame(columns=['speaker_tag', 'text', 'sentiment_score'])

        # fill dataframe, with sentiment_score empty
        for i, text in enumerate(utterances_texts):
                        
            dialogue_df.loc[i] = [utterances_tags[i]] + [text] + ['']

    
    return dialogue_df

