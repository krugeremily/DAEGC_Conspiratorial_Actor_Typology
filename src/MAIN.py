#################### MAIN SCRIPTS TO RUN ALL PRE-PROCESSING AND ANALYSIS SCRIPTS ####################

########## IMPORTS ##########
import argparse
import subprocess
import time
import os
import pandas as pd

########## TIME ##########
start_time = time.time()


########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=bool, default=True , help = 'Should sample be taken (TRUE) or full dataset (FALSE)')
parser.add_argument('--samplesize', type=int, default=200, help = 'Total sample size combined from two datasets')

args = parser.parse_args()

take_sample = args.sample
sample_size = args.samplesize

########## RUN SCRIPTS ##########

#create sample
os.chdir('pre-processing')
subprocess.run(f'python create_sample.py --sample TRUE --samplesize {sample_size}', shell=True)

#convert csv to txt
subprocess.run(f'python pd_to_txt.py --samplesize {sample_size}', shell=True)

#extract count-based features
os.chdir('../analysis')
subprocess.run(f'python feature_extraction.py --samplesize {sample_size}', shell=True)

#run gawk script to for liwc classification
subprocess.run(f'gawk -f liwc_category_ratios.awk ../../data/liwc_german_2007.txt ../../data/samples/messages_sample_{sample_size}.txt > ../../results/liwc_ratios_{sample_size}.csv', shell=True)
print('LIWC classification done.')
########## MERGE RESULTS ##########

os.chdir('..')
#load separate results
print('Merging results.')
ling_features = pd.read_csv(f'../results/messages_with_features_{sample_size}.csv.gzip', compression='gzip').drop('Unnamed: 0', axis=1)
liwc_ratios = pd.read_csv(f'../results/liwc_ratios_{sample_size}.csv', sep=',')
#exclude last column that only contains nan values
liwc_ratios = liwc_ratios.iloc[:, :-1]

#concat liwc_ratios and ling_features based on UID_key
merged = pd.merge(ling_features, liwc_ratios, on='UID_key', how='inner')

#save file
merged.to_csv(f'../results/complete_results_{sample_size}.csv.gzip', compression='gzip', index=False)
print('Complete results saved.')
########## TIME ##########
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
print(f'All run of sample size of {sample_size}. Time taken: {minutes} minutes.')