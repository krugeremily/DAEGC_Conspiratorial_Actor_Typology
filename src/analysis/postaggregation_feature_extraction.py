#################### SCRIPT TO CREATE SAMPLE DF INCLUDING CHANNELS AND GROUPS ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')

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

post_agg = pd.read_csv(f'../results/post-aggregation/liwcANDfeatures_results_{sample_size}.csv.gzip', compression='gzip', index=False)
post_agg['final_message_string'] = post_agg['final_message_string'].astype(str)
#pre_agg['preprocessed_message'] = pre_agg['preprocessed_message'].astype(str)


########## TOXICITY SCORE ##########

#initialize column
post_agg['toxicity'] = 0

#build client
client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

#split df into chunks
n= 10000
list_df = [post_agg[i:i+n] for i in range(0,len(post_agg),n)]

#iterate over chunks and rows to extract toxicity score
final_toxic_list = []
for df in list_df:
    for i in tqdm(range(len(post_agg))):
        row = post_agg.iloc[i]
        toxic = []
        if row['toxicity'] == 0: 
            #split message into list of sentences to pass to toxicity detection function
            tmp = [sent.strip() for sent in re.split(r'[.!?]', row['final_message_string']) if len(sent.split()) > 5]

            if (len(tmp) > 100):
                tmp = random.sample(tmp, 100)
            if (len(tmp) > 1):
                row['toxicity'] = toxicity_detection(tmp, client)

        post_agg.at[i, 'toxicity'] = row['toxicity']

    final_toxic_list.append(df)

#concat chunks
post_agg = pd.concat(final_toxic_list)