import os
import pandas as pd
from Datasets.src.predict_genre import main as predict_genre

TestSub_df = pd.DataFrame(columns = ['movie','genre','subtitles'])
for subdir in os.listdir('Datasets\TestDataset'):
    for file in os.listdir(os.path.join('Datasets\TestDataset', subdir)):
        with open("Datasets/TestDataset/"+subdir+'/'+file, 'r',encoding='Latin-1') as subs:
            data = ' '.join([line for line in subs.readlines()]).replace('\n', ' ')
        TestSub_df = TestSub_df.append({'movie':os.path.splitext(file)[0],'genre':subdir,'subtitles':data},ignore_index=True)
TestSub_df.to_csv('TestSub.csv')
