#################### SCRIPT TO EXTRACT FEATURES FROM MESSAGES ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../utils')
import pandas as pd
pd.set_option('display.max_columns', None)
import regex as re
from utils.linguistic_features import remove_emojis, count_emojis, preprocess_text, count_pos_tags
from textstat import flesch_reading_ease
import time

########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
sample_size = 100000 #how big of a sample to take from each dataset

########## LOAD AND PREPARE DATASET ##########

#load two datasets, drop unnecessary columns and add column to indicate group or channel
groups = pd.read_csv('../data/selected_groups_with_transcriptions.csv.gzip', compression='gzip')
channels = pd.read_csv('../data/channel_subsample.csv.gzip', compression='gzip')

groups = groups.drop(columns=['Unnamed: 0'], axis=1)
groups['group_or_channel'] = 'group'

channels = channels.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
channels['group_or_channel'] = 'channel'


#take random sample of both df where either message or fwd_message contains data and combine
sample_groups = groups[groups['message'].notnull() | groups['fwd_message'].notnull()].sample(n=sample_size, random_state=42)
sample_channels = channels = channels[channels['message'].notnull() | channels['fwd_message'].notnull()].sample(n=sample_size, random_state=42)
combined = pd.concat([sample_groups, sample_channels], ignore_index=True, axis=0)

#keep only necessary columns
messages = combined[['UID_key', 'message', 'fwd_message', 'group_or_channel']]

#remove emojis
cleaned_messages = []
for message in messages['message'].astype(str):
    cleaned_messages.append(remove_emojis(message))

cleaned_fwd_messages = []
for message in messages['fwd_message'].astype(str):
    cleaned_fwd_messages.append(remove_emojis(message))

messages['message_string'] = cleaned_messages
messages['fwd_message_string'] = cleaned_fwd_messages
messages['message_string'] = messages['message_string'].astype(str)
messages['fwd_message_string'] = messages['fwd_message_string'].astype(str)

#if message, take message else take fwd_message
messages['final_message'] = messages['message'].where(messages['message'].notnull(), messages['fwd_message'])
messages['final_message_string'] = messages['message_string'].where(messages['message_string'] != 'nan', messages['fwd_message_string'])

#preprocess text
messages['preprocessed_message'] = messages['final_message_string'].apply(preprocess_text)

#delete uneccessary columns
messages = messages.drop(columns=['message', 'fwd_message', 'message_string', 'fwd_message_string'], axis=1)
messages.to_csv('../data/messages_sample.csv.gzip', compression='gzip')
print('Data loaded and prepared.')

########## SIMPLE COUNT BASED FEATURES ##########

#num sentences
messages['sent_count'] = messages['final_message_string'].apply(lambda x: len(re.split(r'[.!?]+', x)) if x else 0)
#num words
messages['word_count'] = messages['final_message_string'].apply(lambda x: len(re.findall(r'\w+', x)) if x else 0)
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

#use count_pos_tags func to count nouns, verbs and adj
messages['noun_count'] = messages['final_message_string'].apply(lambda x: count_pos_tags(x)[0])
messages['verb_count'] = messages['final_message_string'].apply(lambda x: count_pos_tags(x)[1])
messages['adj_count'] = messages['final_message_string'].apply(lambda x: count_pos_tags(x)[2])
print('Count of selected POS tags extracted.')

########## FLESCH READING EASE SCORE ##########

#use TextStat to compute Flesch Reading Ease score on final_message_string
messages['flesch_reading_ease'] = messages['final_message_string'].apply(flesch_reading_ease)
messages.to_csv('../data/messages_with_features.csv.gzip', compression='gzip')
print('Flesch Reading Ease score extracted.')


########## TIME ##########
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
print(f'Feature extraction done. Runtime: {seconds} seconds (corresponds to {minutes} minutes)')