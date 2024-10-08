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
# author_date = pd.read_csv(f'../../data/aggregated/author_date_{sample_size}.csv.gzip', compression='gzip')
# author_date['final_message_string'] = author_date['final_message_string'].astype(str)
author_group = pd.read_csv(f'../../data/aggregated/author_group_{sample_size}.csv.gzip', compression='gzip')
author_group['final_message_string'] = author_group['final_message_string'].astype(str)
author = pd.read_csv(f'../../data/aggregated/author_{sample_size}.csv.gzip', compression='gzip')
author['final_message_string'] = author['final_message_string'].astype(str)

print('Extracting post-aggregation features from messages.')
########## SET UP TOXICITY API ##########
client = discovery.build(
"commentanalyzer",
"v1alpha1",
developerKey=API_KEY,
discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
static_discovery=False,
)

########## DEFINE COUNT BASED FEATURES ##########
#linguistic feature counts
count_columns = [
    'positive_sentiment',
    'negative_sentiment',
    'neutral_sentiment',
    'channel_messages',
    'group_messages',
]
#for message ratios
message_columns = [
    'own_message',
    'forwarded_message'
    ]

########## ITERATE OVER DATAFRAMES ##########
print('Iterating over dataframes to extract features...')
# for df in tqdm([author_date, author_group, author], desc='Calculating post-aggregation features'):
for df in tqdm([author_group, author], desc='Calculating post-aggregation features'):
    df['own_message_count'] = df['own_message']
    df['forwarded_message_count'] = df['forwarded_message']
    ########## RATIO OF OWN VS. FORWARDED MESSAGES ##########
    for index, row in df.iterrows():
        for col in message_columns:
            df.at[index, col] = row[col] / row['total_message_count']
    ########## CONVERT COUNTS TO PERCENTAGES ##########
    for index, row in df.iterrows():
        for col in count_columns:
            if row['own_message_count'] == 0:
                df.at[index, col] = np.nan
            else:
                df.at[index, col] = row[col] / row['own_message_count']
        ########## ACTION QUOTIENT ##########
        if row['adj_count'] == 0:
            df.at[index, 'action_quotient'] = np.nan
        else:
            df.at[index, 'action_quotient'] = row['verb_count'] / row['adj_count']
        ########## SENTIMENT QUOTIENT ##########
        if row['negative_sentiment'] == 0:
            df.at[index, 'sentiment_quotient'] = np.nan
        else:
            df.at[index, 'sentiment_quotient'] = row['positive_sentiment'] / row['negative_sentiment']

    ########## AVERAGE FLESCH READING EASE SCORE ##########
    #classify scores based on: https://pypi.org/project/textstat/
    df['flesch_reading_ease'] = [x if 0 <= x <= 100 else np.nan for x in df['flesch_reading_ease']]
    flesch_classes = []
    for score in df['flesch_reading_ease']:
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
        
    df['avg_flesch_reading_ease_class'] = flesch_classes

    # ######### TOXICITY SCORE ##########
    
    # split into chunks to stay under quota limit
    n = 9500 
    dfs = [df[i:i+n] for i in range(0, len(df), n)]
    final_df_list = []
    #initialize list for col
    toxicity = []

    #loop over chunks
    for chunk in dfs:
        # get toxicity score from API
        for i in tqdm(range(len(chunk)), desc='Calculating toxicity scores'):
            row = chunk.iloc[i]
            message = row['final_message_string']
            #truncate message to length API can handle
            if len(message) > 500:
                message = message[:500]
            # get toxicity score
            if row['own_message'] == 1:
                tox = toxicity_detection(message, client)
                toxicity.append(tox)
            # for forwarded messages, set to nan
            else:
                toxicity.append(np.nan)
        time.sleep(2)

    # concat all chunks
    df['toxicity'] = toxicity

########## SAVE FILE ##########
# author_date.to_csv(f'../../results/post-aggregation/author_date_{sample_size}.csv.gzip', compression='gzip', index=False)
author_group.to_csv(f'../../results/post-aggregation/author_group_{sample_size}.csv.gzip', compression='gzip', index=False)
author.to_csv(f'../../results/post-aggregation/author_{sample_size}.csv.gzip', compression='gzip', index=False)
print('Post aggregation results saved.')

########## TIME ##########
end_time = time.time()
print(f'Post-aggregation feature extraction done on sample size of {sample_size} in {(end_time - start_time)/60} minutes.')