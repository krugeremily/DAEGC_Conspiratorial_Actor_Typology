#################### SCRIPT TO REFORMAT ORIGINAL LIWC DICT FILE ####################

########## IMPORTS ##########
import pandas as pd

########## LOAD ORIGINAL LIWC DICT ##########

file_path = '../data/LIWC2007_German.dic'
#the first 70 rows should not be read as they contain metadata
skiprows = 70

data = []
with open(file_path, 'r', encoding='latin1') as file:
    #skip metadata rows
    for _ in range(skiprows):
        next(file)
    
    #read file line-by-line
    for line in file:
        split_line = line.strip().split('\t')
        word = split_line[0]
        categories = split_line[1:]
        data.append([word, categories])

#create df
headers = ['word', 'categories']
df = pd.DataFrame(data, columns=headers)

########## EXPAND CATEGORIES ##########
df = df.explode('categories')
df['categories'] = df['categories'].astype(int)


########## MAP CATEGORY NAMES ##########
#dict based on metadata at beginning of original LIWC dict file
liwc_categories = {
    1: 'Pronoun',
    2: 'I',
    3: 'We',
    4: 'Self',
    5: 'You',
    6: 'Other',
    7: 'Negate',
    8: 'Assent',
    9: 'Article',
    10: 'Preps',
    11: 'Number',
    12: 'Affect',
    13: 'Posemo',
    14: 'Posfeel',
    15: 'Optim',
    16: 'Negemo',
    17: 'Anx',
    18: 'Anger',
    19: 'Sad',
    20: 'Cogmech',
    21: 'Cause',
    22: 'Insight',
    23: 'Discrep',
    24: 'Inhib',
    25: 'Tentat',
    26: 'Certain',
    27: 'Senses',
    28: 'See',
    29: 'Hear',
    30: 'Feel',
    31: 'Social',
    32: 'Comm',
    33: 'Othref',
    34: 'Friends',
    35: 'Family',
    36: 'Humans',
    37: 'Time',
    38: 'Past',
    39: 'Present',
    40: 'Future',
    41: 'Space',
    42: 'Up',
    43: 'Down',
    44: 'Incl',
    45: 'Excl',
    46: 'Motion',
    47: 'Occup',
    48: 'School',
    49: 'Job',
    50: 'Achieve',
    51: 'Leisure',
    52: 'Home',
    53: 'Sports',
    54: 'TV',
    55: 'Music',
    56: 'Money',
    57: 'Metaph',
    58: 'Relig',
    59: 'Death',
    60: 'Physcal',
    61: 'Body',
    62: 'Sexual',
    63: 'Eating',
    64: 'Sleep',
    65: 'Groom',
    66: 'Swear',
    67: 'Nonfl',
    68: 'Fillers'
}

#map category names
df['cat_name'] = df['categories'].map(liwc_categories)
df = df[['word', 'cat_name', 'categories']]

#save reformattet dict
df.to_csv('../data/liwc_german_2007.txt', sep='\t', index=False, header=False)