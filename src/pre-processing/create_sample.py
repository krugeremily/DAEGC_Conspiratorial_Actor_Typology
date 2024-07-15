#################### SCRIPT TO CREATE SAMPLE DF INCLUDING CHANNELS AND GROUPS ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../utils')

import time
import argparse
import pandas as pd
from utils.linguistic_features import remove_emojis, remove_tags, preprocess_text
from tqdm import tqdm


########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=int, help = 'Sample taken from EACH of the two datasets')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

sample_size = args.sample #how big of a sample to take from each dataset
random_state = args.seed

########## LOAD AND PREPARE DATASET ##########

#load two datasets, drop unnecessary columns and add column to indicate group or channel
groups = pd.read_csv('../data/selected_groups_with_transcriptions.csv.gzip', compression='gzip').drop(columns=['Unnamed: 0'], axis=1)
channels = pd.read_csv('../data/channel_subsample.csv.gzip', compression='gzip').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)


groups['group_or_channel'] = 'group'
channels['group_or_channel'] = 'channel'


#take random sample of both df where either message or fwd_message contains data and combine
sample_groups = groups[groups['message'].notnull() | groups['fwd_message'].notnull()].sample(n=sample_size, random_state=random_state)
sample_channels = channels = channels[channels['message'].notnull() | channels['fwd_message'].notnull()].sample(n=sample_size, random_state=random_state)
combined = pd.concat([sample_groups, sample_channels], ignore_index=True, axis=0)

#keep only necessary columns
messages = combined[['UID_key', 'message', 'fwd_message', 'group_or_channel']]

#remove emojis and links
cleaned_messages = []
for message in messages['message'].astype(str):
    message = remove_tags(message)
    cleaned_messages.append(remove_emojis(message))

cleaned_fwd_messages = []
for message in messages['fwd_message'].astype(str):
    message = remove_tags(message)
    cleaned_fwd_messages.append(remove_emojis(message))

messages['message_string'] = cleaned_messages
messages['fwd_message_string'] = cleaned_fwd_messages
messages['message_string'] = messages['message_string'].astype(str)
messages['fwd_message_string'] = messages['fwd_message_string'].astype(str)

#if message, take message else take fwd_message
messages['final_message'] = messages['message'].where(messages['message'].notnull(), messages['fwd_message']).astype(str)
messages['final_message_string'] = messages['message_string'].where(messages['message_string'] != 'nan', messages['fwd_message_string']).astype(str)

#preprocess text
preprocessed_messages = []
for message in tqdm(messages['final_message_string'], desc = 'Preprocessing messages'):
    message = preprocess_text(message)
    preprocessed_messages.append(message)
messages['preprocessed_message'] = preprocessed_messages

#delete uneccessary columns
messages = messages.drop(columns=['message', 'fwd_message', 'message_string', 'fwd_message_string'], axis=1)

os.makedirs('../data/samples', exist_ok=True)
messages.to_csv(f'../data/samples/messages_sample_{sample_size*2}.csv.gzip', compression='gzip')
print('')

########## TIME ##########
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
print(f'Sample of {sample_size*2 }created and saved.Time taken: {minutes} minutes.')