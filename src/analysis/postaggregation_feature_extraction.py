#################### SCRIPT TO CREATE SAMPLE DF INCLUDING CHANNELS AND GROUPS ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../../')

from functions.linguistic_features import toxicity_detection
import time
import argparse
import pandas as pd
import numpy as np
import regex as re
from tqdm import tqdm
import random

from googleapiclient import discovery
from config import API_KEY

########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='100', help = 'Total sample size combined from two datasets as int or "full"')
args = parser.parse_args()

sample_size = args.samplesize


########## LOAD DATASET ##########
print('Loading datasets...')
author_date = pd.read_csv(f'../../data/aggregated/author_date_{sample_size}.csv.gzip', compression='gzip')
author_date['final_message_string'] = author_date['final_message_string'].astype(str)
author_group = pd.read_csv(f'../../data/aggregated/author_group_{sample_size}.csv.gzip', compression='gzip')
author_group['final_message_string'] = author_group['final_message_string'].astype(str)

########## SET UP TOXICITY API ##########
client = discovery.build(
"commentanalyzer",
"v1alpha1",
developerKey=API_KEY,
discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
static_discovery=False,
)

########## DEFINE COUNT BASED FEATURES ##########
count_columns = [
    'own_message',
    'forwarded_message',
    'positive_sentiment',
    'negative_sentiment',
    'neutral_sentiment',
    'channel_messages',
    'group_messages',
    'flesch_reading_ease_class_difficult',
    'flesch_reading_ease_class_easy',
    'flesch_reading_ease_class_fairly difficult',
    'flesch_reading_ease_class_fairly easy',
    'flesch_reading_ease_class_standard',
    'flesch_reading_ease_class_unclassified',
    'flesch_reading_ease_class_very confusing',
    'flesch_reading_ease_class_very easy'
]

########## ITERATE OVER DATAFRAMES ##########
print('Iterating over dataframes to extract features...')
for df in tqdm([author_date, author_group], desc='Calculating post-aggregation features'):
    ########## CONVERT COUNTS TO PERCENTAGES ##########
    for index, row in df.iterrows():
        for col in count_columns:
            df.at[index, col] = row[col] / row['message_count']

    ########## ACTION QUOTIENT ##########
    df['action_quotient'] = df['verb_count'] / df['adj_count']

    ########## TOXICITY SCORE ##########

    #initialize column
    df['toxicity'] = 0

    #split df into chunks
    n= 10000
    list_df = [df[i:i+n] for i in range(0,len(df),n)]

    #iterate over chunks and rows to extract toxicity score
    final_toxic_list = []
    for df in list_df:
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            toxic = []
            if row['toxicity'] == 0: 
                #split message into list of sentences to pass to toxicity detection function
                tmp = [sent.strip() for sent in re.split(r'[.!?]', row['final_message_string']) if len(sent.split()) > 5]

                if (len(tmp) > 100):
                    tmp = random.sample(tmp, 100)
                if (len(tmp) > 1):
                    row['toxicity'] = toxicity_detection(tmp, client)

            df.at[i, 'toxicity'] = row['toxicity']

        final_toxic_list.append(df)

    #concat chunks
    df = pd.concat(final_toxic_list)

########## SAVE FILE ##########
author_date.to_csv(f'../../results/post-aggregation/author_date_{sample_size}.csv.gzip', compression='gzip', index=False)
author_group.to_csv(f'../../results/post-aggregation/author_group_{sample_size}.csv.gzip', compression='gzip', index=False)
print('Post aggregation results saved.')

########## TIME ##########
end_time = time.time()
print(f'Post-aggregation feature extraction done in {(end_time - start_time)/60} minutes.')