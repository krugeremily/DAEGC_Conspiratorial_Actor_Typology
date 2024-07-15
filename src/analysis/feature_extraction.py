#################### SCRIPT TO EXTRACT FEATURES FROM MESSAGES ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')

import time
import argparse
import pandas as pd
import regex as re
import numpy as np
from functions.linguistic_features import count_emojis, count_pos_tags
import textstat
from tqdm import tqdm

########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=int, help = 'Total sample size of prepared dataset (as in filename)')
args = parser.parse_args()

sample_size = args.samplesize #sample size of loaded dataset

########## LOAD DATASET ##########

messages = pd.read_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip').drop(columns=['Unnamed: 0'], axis=1)
messages['final_message_string'] = messages['final_message_string'].astype(str)
messages['preprocessed_message'] = messages['preprocessed_message'].astype(str)

########## SIMPLE COUNT BASED FEATURES ##########

#num sentences
messages['sent_count'] = messages['final_message_string'].apply(lambda x: len(re.split(r'[.!?]+', x)) if x != '' else 0)
#num words
messages['word_count'] = messages['final_message_string'].apply(lambda x: len(re.findall(r'\w+', x)) if x != '' else 0)
#avg sentence length (words per sentence)
messages['avg_sent_length'] = messages.apply(lambda row: row['word_count'] / row['sent_count'] if row['sent_count'] > 0 else 0, axis=1)
#avg word length (characters per word)
messages['avg_word_length'] = messages.apply(lambda row: len(row['final_message_string'].replace(' ', '')) / row['word_count'] if row['word_count'] > 0 else 0, axis=1)
#num exclamations (multiple ! coutn as one exclamation)
messages['exclamation_count'] = messages['final_message_string'].apply(lambda x: len(re.findall(r'!+', x)) if x else 0)
#num questions (multiple ? count as one question)
messages['question_count'] = messages['final_message_string'].apply(lambda x: len(re.findall(r'\?+', x)) if x else 0)
#num emojis 
messages['emoji_count'] = messages['final_message'].apply(lambda x: count_emojis(x) if x else 0)

print('Simple count based features extracted.')

########## COUNT OF SELECTED POS TAGS ##########

#count nouns, verbs and adj
nouns = []
verbs = []
adjectives = []

for message in tqdm(messages['final_message_string'], desc = 'Extracting POS Tag counts'):
    noun, verb, adj = count_pos_tags(message)
    nouns.append(noun)
    verbs.append(verb)
    adjectives.append(adj)
                    
messages['noun_count'] = nouns
messages['verb_count'] = verbs
messages['adj_count'] = adjectives

########## FLESCH READING EASE SCORE ##########

textstat.set_lang('de')
#compute Flesch Reading Ease score on non-empty messages
messages['flesch_reading_ease'] = messages['final_message_string'].apply(lambda x: textstat.flesch_reading_ease(x) if x.strip() != '' and x != 'nan' else np.nan)

#classify scores based on: https://pypi.org/project/textstat/
flesch_classes = []
for score in messages['flesch_reading_ease']:
    if score >= 0 and score < 30:
        flesch_classes.append('very confusing')
    elif score >= 30 and score < 50:
        flesch_classes.append('difficult')
    elif score >= 50 and score < 60:
        flesch_classes.append('fairly difficult')
    elif score >=60 and score < 70:
        flesch_classes.append('standard')
    elif score >=70 and score < 80:
        flesch_classes.append('fairly easy')
    elif score >=80 and score < 90:
        flesch_classes.append('easy')
    elif score >=90 and score < 101:
        flesch_classes.append('very easy')
    else:
        flesch_classes.append('unclassified')
    
messages['flesch_reading_ease_class'] = flesch_classes

print('Flesch Reading Ease score extracted.')

os.makedirs('../../results', exist_ok=True)
messages.to_csv(f'../../results/messages_with_features_{sample_size}.csv.gzip', compression='gzip')

########## TIME ##########
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
print(f'Feature extraction done. Runtime: {seconds} seconds (corresponds to {minutes} minutes) for sample of {sample_size}')