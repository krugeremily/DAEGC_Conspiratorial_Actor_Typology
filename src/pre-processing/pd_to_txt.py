#################### SCRIPT TO FORMAT CSV DATAFRAME TO TXT FILE FOR GAWL SCRIPT####################

########## IMPORTS ##########
import pandas as pd
import argparse


########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=int, default=100, help = 'Total samplesize')
args = parser.parse_args()

sample_size = args.samplesize 

########## LOAD DATASET ##########
data = pd.read_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip').drop('Unnamed: 0', axis=1)

########## FORMAT TO TXT ##########

#keep only necessary columns
data = data[['UID_key', 'final_message_string']]
data.to_csv(f'../../data/samples/messages_sample_{sample_size}.txt', sep='\t', index=False, header=False, quoting=3)
print('Dataframe formatted to txt.')