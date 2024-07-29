#################### SCRIPT TO AGGREGATE MESSAGES AND METRICS PER AUTHOR, GROUP AND MONTH ####################

########## IMPORTS ##########
import pandas as pd
import time
import argparse

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

pre_agg = pd.get_dummies(pre_agg, columns=['group_or_channel', 'flesch_reading_ease_class'])
print('Dummies for categorial variables created.')

########## DEFINE AGGREGATION DICT ##########
# Aggregation dictionary
agg_dict = {
    # COUNT
    'UID_key': 'count',

    # SUM
    'own_message': 'sum',
    'forwarded_message': 'sum',
    'noun_count': 'sum',
    'verb_count': 'sum',
    'adj_count': 'sum',
    'positive_sentiment': 'sum',
    'negative_sentiment': 'sum',
    'neutral_sentiment': 'sum',
    'group_or_channel_channel': 'sum',
    'group_or_channel_group': 'sum',
    'flesch_reading_ease_class_difficult': 'sum',
    'flesch_reading_ease_class_easy': 'sum',
    'flesch_reading_ease_class_fairly difficult': 'sum',
    'flesch_reading_ease_class_fairly easy': 'sum',
    'flesch_reading_ease_class_standard': 'sum',
    'flesch_reading_ease_class_unclassified': 'sum',
    'flesch_reading_ease_class_very confusing': 'sum',
    'flesch_reading_ease_class_very easy': 'sum',

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
    'fwd_message': lambda x: ' '.join(x.dropna().astype(str)),
    'fwd_message_string': lambda x: ' '.join(x.dropna().astype(str)),
    'final_message': lambda x: ' '.join(x.dropna().astype(str)),
    'final_message_string': lambda x: ' '.join(x.dropna().astype(str)),
}

########## RENAMING COLUMNS ##########

rename_dict = {'group_or_channel_channel': 'channel_messages', 'group_or_channel_group': 'group_messages', 'UID_key': 'message_count'}

########## AGGREGATE PER AUTHOR&MONTH ##########

print('Aggregating per author and month...')
agg_author_date = pre_agg.groupby(['author', 'date']).agg(agg_dict)
agg_author_date = agg_author_date.rename(columns=rename_dict)
agg_author_date.to_csv(f'../../data/aggregated/author_date_{sample_size}.csv.gzip', compression='gzip')
print('Aggregation per author and month complete.')

########## AGGREGATE PER AUTHOR&GROUP ##########

print('Aggregating per author and group...')
agg_author_group = pre_agg.groupby(['author', 'group_name']).agg(agg_dict)
agg_author_group = agg_author_group.rename(columns=rename_dict)
agg_author_group.to_csv(f'../../data/aggregated/author_group_{sample_size}.csv.gzip', compression='gzip')
print('Aggregation per author and group complete.')

########## AGGREGATE PER AUTHOR ##########

print('Aggregating per author...')
agg_author = pre_agg.groupby(['author']).agg(agg_dict)
agg_author = agg_author.rename(columns=rename_dict)
agg_author.to_csv(f'../../data/aggregated/author_{sample_size}.csv.gzip', compression='gzip')
print('Aggregation per author complete.')

########## TIME ##########
end_time = time.time()
print(f'Aggregation complete in {(end_time - start_time)} seconds.')