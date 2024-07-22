#################### SCRIPT TO CREATE SAMPLE DF INCLUDING CHANNELS AND GROUPS ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')

import time
import argparse
import pandas as pd
import numpy as np
from functions.linguistic_features import remove_emojis, remove_tags, preprocess_text
from tqdm import tqdm


########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='100', help = 'Total sample size combined from two datasets as int or "full"')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
args = parser.parse_args()

sample_size = args.samplesize
random_state = args.seed

########## LOAD AND PREPARE DATASET ##########

#load two datasets, drop unnecessary columns and add column to indicate group or channel
print('Loading datsets.')
groups = pd.read_csv('../../data/selected_groups_with_transcriptions.csv.gzip', compression='gzip').drop(columns=['Unnamed: 0'], axis=1)
channels = pd.read_csv('../../data/channel_subsample.csv.gzip', compression='gzip').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)


groups['group_or_channel'] = 'group'
channels['group_or_channel'] = 'channel'


#if desired, take random sample of both df where either message or fwd_message (or transcribed_messgae if group) contains data and combine
if sample_size != 'full':
    print('Taking samples.')
    sample_each = int(int(sample_size) / 2)
    sample_groups = groups[groups['message'].notnull() | groups['fwd_message'].notnull() | groups['transcribed_message'].notnull()].sample(n=sample_each, random_state=random_state)
    sample_channels = channels[channels['message'].notnull() | channels['fwd_message'].notnull()].sample(n=sample_each, random_state=random_state)
    combined = pd.concat([sample_groups, sample_channels], ignore_index=True, axis=0)
else:
    combined = pd.concat([groups, channels], ignore_index=True, axis=0)

#keep only necessary columns
messages = combined[['UID_key','author', 'message', 'fwd_message', 'transcribed_message', 'group_or_channel']]

#remove emojis and links
print('Cleaning messages.')
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


print('Combining the three message columns into one.')
#if message, take message else take fwd_message else take transcribed message
messages['final_message'] = messages['message'].combine_first(messages['fwd_message']).combine_first(messages['transcribed_message']).astype(str)
# Compute the 'final_message_string' column
messages['final_message_string'] = messages['message_string'].replace('nan', np.nan).combine_first(
                                   messages['fwd_message_string'].replace('nan', np.nan)).combine_first(
                                   messages['transcribed_message'].replace('nan', np.nan)).astype(str)

#in final_message_string, replace multiple whitespaces with one
messages['final_message_string'] = messages['final_message_string'].str.replace(r'\s+', ' ', regex=True)

# #preprocess text
# preprocessed_messages = []
# for message in tqdm(messages['final_message_string'], desc = 'Preprocessing messages'):
#     message = preprocess_text(message)
#     preprocessed_messages.append(message)
# messages['preprocessed_message'] = preprocessed_messages

#delete uneccessary columns
messages = messages.drop(columns=['message', 'fwd_message', 'message_string', 'fwd_message_string', 'transcribed_message'], axis=1)
print('Saving dataset.')
os.makedirs('../../data/samples', exist_ok=True)
messages.to_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip')

########## TIME ##########
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
print(f'Sample of {sample_size} created and saved. Time taken: {minutes} minutes.')