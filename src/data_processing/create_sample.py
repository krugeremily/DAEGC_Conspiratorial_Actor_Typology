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
print('Imports done.')


########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='200', help = 'Total sample size combined from two datasets as int or "full"')
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

groups = groups[groups['message'].notnull() | groups['transcribed_message'].notnull()]
channels = channels[channels['message'].notnull()]

group_len = len(groups)
channel_len = len(channels)


#if desired, take random sample of both df where either message or fwd_message (or transcribed_messgae if group) contains data and combine
if sample_size == 'full':
    combined = pd.concat([groups, channels], ignore_index=True, axis=0) 
else:
    print('Taking samples.')
    #try taking same amount of rows each
    samplesize_group = int(int(sample_size) / 2)
    samplesize_channel = int(int(sample_size) / 2)
    #if not enough rows in channel, take full and rest from groups
    if samplesize_channel > channel_len:
        print('Sample size too large for channel. Taking full channel dataset.')
        samplesize_channel = channel_len
        samplesize_group = int(sample_size) - samplesize_channel
    #if not enough rows in either, take both full datasets
    if samplesize_group > group_len:
        print('Sample size too large for group & channels. Taking full datasets.')
        combined = pd.concat([groups, channels], ignore_index=True, axis=0)
    else:
        sample_groups = groups[groups['message'].notnull() | groups['transcribed_message'].notnull()].sample(n=samplesize_group, random_state=random_state)
        sample_channels = channels[channels['message'].notnull()].sample(n=samplesize_channel, random_state=random_state)
        combined = pd.concat([sample_groups, sample_channels], ignore_index=True, axis=0)



#make date column for aggregation
combined['date'] = combined.apply(lambda row: f"{str(row['year'])}-{str(row['month'])}", axis=1)

#for counting own and transcribed messages
combined['own_message'] = [1 if x else 0 for x in combined['message'].notnull()]
combined['forwarded_message'] = [1 if x else 0 for x in combined['fwd_message'].notnull()]

#keep only necessary columns
messages = combined[['UID_key','author', 'message', 'date', 'transcribed_message', 'group_or_channel', 'own_message', 'forwarded_message']]


#remove emojis and links
print('Cleaning messages.')
cleaned_messages = []
for message in messages['message'].astype(str):
    message = remove_tags(message)
    cleaned_messages.append(remove_emojis(message))


messages['message_string'] = cleaned_messages
messages['message_string'] = messages['message_string'].astype(str)


print('Combining the two message columns into one.')
#if message, take message else take transcribed message
messages['final_message'] = messages['message'].combine_first(messages['transcribed_message']).astype(str)
# Compute the 'final_message_string' column
messages['final_message_string'] = messages['message_string'].replace('nan', np.nan).combine_first(
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
messages = messages.drop(columns=['message', 'message_string', 'transcribed_message'], axis=1)
print('Saving dataset.')
os.makedirs('../../data/samples', exist_ok=True)
messages.to_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip')

########## TIME ##########
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
print(f'Sample of {sample_size} created and saved. Time taken: {minutes} minutes.')