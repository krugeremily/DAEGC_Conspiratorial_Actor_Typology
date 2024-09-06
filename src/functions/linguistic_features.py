import regex as re
import spacy
import json

################## HANDLING EMOJIS ##################

#remove emojis from messages (function found on stackoverflow)
#https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def remove_emojis(text):
    emoj = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002500-\U00002BEF'  # chinese char
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642' 
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
                      ']+', re.UNICODE)
    return re.sub(emoj, '', text)

#find number of emojis in messages
def count_emojis(text):
    emoj = re.compile('['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002500-\U00002BEF'  # chinese char
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642' 
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
                      ']+', re.UNICODE)
    return len(re.findall(emoj, text))

################## REMOVING LINKS, URLS AND OTHER TAGS ##################

def remove_tags(text):
    #remove everything within a tag
    a_tag_pattern = r'<a href=".*?">.*?</a>'
    custom_tag_pattern = r'<\w+>|<\/\w+>'
    cleaned_text = re.sub(a_tag_pattern, '', text)
    cleaned_text = re.sub(custom_tag_pattern, '', cleaned_text)
    return cleaned_text

################## PREPROCESSING & POS TAGGING WITH SPACY ##################

nlp = spacy.load('de_core_news_sm', disable=['parser', 'ner'])

def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    message = ' '.join(tokens)
    return message


def count_pos_tags(text):
    global nlp
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    nouns = pos_tags.count('NOUN')
    verbs = pos_tags.count('VERB')
    adjectives = pos_tags.count('ADJ')
    return nouns, verbs, adjectives

################## TOXICITY SCORE VIA PERSPECTIVE API ##################

def toxicity_detection(sentences, client):
    toxic = []
    for sent in sentences:
        analyze_request = {
            'comment': { 'text': f"{sent}" },
            'languages' : ["de"],
            'requestedAttributes': {'TOXICITY': {}},
        }

        response = client.comments().analyze(body=analyze_request).execute()
        j = json.dumps(response, indent=2)
        #print(json.loads(j)['attributeScores']['TOXICITY']['summaryScore']['value'])
        toxic.append(json.loads(j)['attributeScores']['TOXICITY']['summaryScore']['value'])
    avg = sum(toxic)/len(toxic)
    #print(avg)
    return avg