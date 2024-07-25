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