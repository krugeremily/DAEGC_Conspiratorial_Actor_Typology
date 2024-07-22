import pandas as pd
import time
from germansentiment import SentimentModel

messages = pd.read_csv(f'../../data/samples/messages_sample_200000.csv.gzip', compression='gzip').drop(columns=['Unnamed: 0'], axis=1)
messages['final_message_string'] = messages['final_message_string'].fillna('')
messages['final_message_string'] = messages['final_message_string'].astype(str)
messages['preprocessed_message'] = messages['preprocessed_message'].astype(str)

model = SentimentModel()

#no probabilities
print('Predicting sentiment without probabilities...')
no_probs_start = time.time()
messages['sentiment'] = model.predict_sentiment(messages['final_message_string'])
no_probs_end = time.time()
no_probs_time = (no_probs_end - no_probs_start) /60
print(f'Predicting sentiment without probabilities took {no_probs_time} minutes.')

#with probabilities
print('Predicting sentiment with probabilities...')
with_probs_start = time.time()
messages['sentiment_'], messages['probs'] = model.predict_sentiment(messages['final_message_string'], output_probabilities=True)
messages[['positive_prob', 'negative_prob', 'neutral_prob']] = pd.DataFrame(messages['probs'].tolist(), index=messages.index).applymap(lambda x: x[1])
with_probs_end = time.time()
with_probs_time = (with_probs_end - with_probs_start) /60
print(f'Predicting sentiment with probabilities took {with_probs_time} minutes.')