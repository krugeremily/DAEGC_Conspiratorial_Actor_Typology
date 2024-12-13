#################### SCRIPT TO AGGREGATE MESSAGES AND METRICS PER AUTHOR, GROUP AND MONTH ####################

########## IMPORTS ##########
import pandas as pd
import time
import argparse
import os

########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='100', help = 'Total sample size combined from two datasets as int or "full"')
args = parser.parse_args()

sample_size = args.samplesize #sample size of loaded dataset

########## LOAD DATASET ##########
pre_agg = pd.read_csv(f'../../results/pre-aggregation/liwcANDfeatures_results_{sample_size}.csv.gzip', compression='gzip')
print('Dataset loaded.')

pre_agg = pd.get_dummies(pre_agg, columns=['group_or_channel'])
print('Dummies for categorial variables created.')

########## SAMPLE TO CALCULATE MESSAGE RATIOS ##########
"""
Calculating the ratio of own vs. forwarded messages has to be calculated separately and before the aggregation of other features.
As the linguistic features are only calculate on own messages, and forearded messages are assigned nan or 0 values, including them in the ratio would scew results.
Instead the ratios will be calculated separately and added to the aggregated dataframes afterwards.
"""

messages = pre_agg[['author', 'own_message', 'forwarded_message', 'fwd_author', 'UID_key', 'group_name', 'date']]
pre_agg = pre_agg[pre_agg['own_message'] == 1]

########## DEFINE AGGREGATION DICT ##########
# Aggregation dictionary for linguistic features
agg_dict = {
    # SUM
    'noun_count': 'sum',
    'verb_count': 'sum',
    'adj_count': 'sum',
    'positive_sentiment': 'sum',
    'negative_sentiment': 'sum',
    'neutral_sentiment': 'sum',
    'group_or_channel_channel': 'sum',
    'group_or_channel_group': 'sum',

    # AVG
    'sent_count': 'mean',
    'word_count': 'mean',
    'avg_sent_length': 'mean',
    'avg_word_length': 'mean',
    'exclamation_count': 'mean',
    'question_count': 'mean',
    'emoji_count': 'mean',
    'flesch_reading_ease': 'mean',
    'liwc_I': 'mean',
    'liwc_We': 'mean',
    'liwc_You': 'mean',
    'liwc_Other': 'mean',
    'liwc_Affect': 'mean',
    
    # ' '.JOIN
    'final_message': lambda x: ' '.join(x.dropna().astype(str)),
    'final_message_string': lambda x: ' '.join(x.dropna().astype(str)),
}

# Aggregation dictionary for message ratios
agg_dict_messages = {
    'own_message': 'sum',
    'forwarded_message': 'sum',
    'UID_key': 'count'
}

########## RENAMING COLUMNS ##########

rename_dict = {'group_or_channel_channel': 'channel_messages', 'group_or_channel_group': 'group_messages', 'UID_key': 'total_message_count'}

########## AGGREGATE PER AUTHOR&MONTH ##########

os.makedirs('../../data/aggregated', exist_ok = True)

# print('Aggregating per author and month...')

# #aggregate linguistic features
# agg_author_date = pre_agg.groupby(['author', 'date']).agg(agg_dict)
# agg_author_date = agg_author_date.rename(columns=rename_dict)
# #aggregate message ratios
# agg_author_date_messages = messages.groupby(['author', 'date']).agg(agg_dict_messages)
# agg_author_date_messages = agg_author_date_messages.rename(columns=rename_dict)
# #concat based on author and date columns
# agg_author_date = pd.merge(
#     left = agg_author_date,
#     right = agg_author_date_messages,
#     how = 'outer',
#     left_on = ['author', 'date'],
#     right_on = ['author', 'date']
# )

# # measure for how often author was forwarded
# for author,date in agg_author_date.index:
#     # count how often message by this author was forwarded in this group
#     was_forwarded_author_date = len(messages[(messages['fwd_author'] == author) & (messages['date'] == date)])
# agg_author_date['was_forwarded'] = was_forwarded_author_date

# #save to csv
# agg_author_date.to_csv(f'../../data/aggregated/author_date_{sample_size}.csv.gzip', compression='gzip')
# print('Aggregation per author and month complete.')

########## AGGREGATE PER AUTHOR&GROUP ##########

print('Aggregating per author and group...')
#aggregate linguistic features
agg_author_group = pre_agg.groupby(['author', 'group_name']).agg(agg_dict)
agg_author_group = agg_author_group.rename(columns=rename_dict)
#aggregate message ratios
agg_author_group_messages = messages.groupby(['author', 'group_name']).agg(agg_dict_messages)
agg_author_group_messages = agg_author_group_messages.rename(columns=rename_dict)
#concat based on author and group columns
agg_author_group = pd.merge(
    left = agg_author_group,
    right = agg_author_group_messages,
    how = 'outer',
    left_on = ['author', 'group_name'],
    right_on = ['author', 'group_name']
)

# measure for how often author was forwarded
was_forwarded_author_group = []
for author,group in agg_author_group.index:
    # count how often message by this author was forwarded in this group
    fwd = len(messages[(messages['fwd_author'] == author) & (messages['group_name'] == group)])
    was_forwarded_author_group.append(fwd)
agg_author_group['was_forwarded'] = was_forwarded_author_group

#save to csv
agg_author_group.to_csv(f'../../data/aggregated/author_group_{sample_size}.csv.gzip', compression='gzip')
print('Aggregation per author and group complete.')

########## AGGREGATE PER AUTHOR ##########
print('Aggregating per author...')
#aggregate linguistic features
agg_author = pre_agg.groupby(['author']).agg(agg_dict)
agg_author = agg_author.rename(columns=rename_dict)
#aggregate message ratios
agg_author_messages = messages.groupby(['author']).agg(agg_dict_messages)
agg_author_messages = agg_author_messages.rename(columns=rename_dict)
#concat based on author column
agg_author = pd.merge(
    left = agg_author,
    right = agg_author_messages,
    how = 'outer',
    left_on = ['author'],
    right_on = ['author']
)

# measure for how often author was forwarded
was_forwarded_author = []
for author in agg_author.index:
    # count how often message by this author was forwarded in this group
    fwd = len(messages[(messages['fwd_author'] == author)])
    was_forwarded_author.append(fwd)
agg_author['was_forwarded'] = was_forwarded_author

#save to csv
agg_author.to_csv(f'../../data/aggregated/author_{sample_size}.csv.gzip', compression='gzip')
print('Aggregation per author complete.')

########## TIME ##########
end_time = time.time()
print(f'Aggregation complete in {(end_time - start_time)} seconds.')