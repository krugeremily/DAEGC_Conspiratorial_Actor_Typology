https://docs.google.com/document/d/1B9S35aB3A0MxObnLSVdA7QrruzmPaArz_OkyyoquPr8/edit?pli=1

# Running Tasks

>[!Running Tasks]
>- [x] Find further linguistic features and a way to extract them
>- [ ] look into emoji classifier/ emoji use for sarcasm detection
>- [ ] look into German LWIC: how to use, license needed etc
>- [ ] write code to enrich dataset with:
> 	 - [x] count-based features
> 	 - [x] POS tagging
> 	 - [ ] emoji classifier
> 	 - [ ] language complexity
> 		 - [ ] hugging face model
> 		 - [x] textstat Flesch readability ease score
> 	 - [ ] sentiment: German LIWC
> 	 - [ ] narrative novelty vs. continuity
>- [ ] get estimate for runtime (on ~200.000 rows)
>- [ ] run on full dataset
>
# Monday, 24/06/2024

> [!Tasks]
> - [x] running task: Find further linguistic features and a way to extract them
> - [x] Understand the dataset and each column

## Understanding Dataset Columns

**Dataset: selected_groups_with_transcriptions.csv.gzip**

| Index | Column                   | Type            | Content                                                                |     |
| ----- | ------------------------ | --------------- | ---------------------------------------------------------------------- | --- |
| 0     | UID_key                  |                 | unique ID: hashed group name + message ID                              |     |
| 1     | initial_ID               | int             |                                                                        |     |
| 2     | mid_message              | float           | ID of message in respective group                                      |     |
| 3     | mid_file                 | float           | ID of media file in respective group                                   |     |
| 4     | group_name               | string + emojis | name of the group/channel                                              |     |
| 5     | posting_date             |                 | date and time of message in the group                                  |     |
| 6     | message                  | string + emojis | message content                                                        |     |
| 7     | fwd_message              | float           | forwarded message content                                              |     |
| 8     | fwd_posting_date_message | float           | for forwarded messages: date and time of message in the original group |     |
| 9     | posting_date_file        |                 | for files: date and time of file in the group                          |     |
| 10    | link_url                 |                 | url as found in HTML a-tag                                             |     |
| 11    | media_file               |                 | name of media file                                                     |     |
| 12    | media_file_type          |                 | voice_message, video, or photo                                         |     |
| 13    | fwd_posting_date_file    |                 | for forwarded files: date and time of file in the original group       |     |
| 14    | fwd_link_url             |                 | for forwarded messages: url as found in HTML a-tag in original message |     |
| 15    | fwd_media_file           |                 | for forwarded messages: name of forwarded media file                   |     |
| 16    | fwd_media_file_type      |                 | for forwarded messages: voice_message, video, or photo                 |     |
| 17    | author                   | float           | hashed name for author of message                                      |     |
| 18    | fwd_author               | float           | for forwarded messages: hashed name for author of original message     |     |
| 19    | day                      | float           | day of a month (1-31)                                                  |     |
| 20    | week                     | float           | week of year (1-52)                                                    |     |
| 21    | weekday                  | float           | day of week (1-7)                                                      |     |
| 22    | message_hash             | float           | hashed message content                                                 |     |
| 23    | fwd_message_hash         | float           | for forwarded messages: hashed message content of forwarded message    |     |
| 24    | website                  |                 | website domain, if link_url refers to a website                        |     |
| 25    | replied_to               |                 | message_id of message that has been replied to                         |     |
| 26    | year                     | float           | year                                                                   |     |
| 27    | month                    | float           | month of year (1-12)                                                   |     |
| 28    | day_of_year              | float           | day of a year (1-366)                                                  |     |
| 29    | duration                 | float           | duration of media file (in seconds?)                                   |     |
| 30    | filepath                 |                 | filepath to media file                                                 |     |
| 31    | filename                 |                 | same as media_file                                                     |     |
| 32    | filename_if_joined       |                 | for joining transcription dataset                                      |     |
| 33    | transcribed_message      |                 | transcription of media file                                            |     |
| 34    | newsguard_domain         |                 | newsguard website domain                                               |     |
| 35    | newsguard_score          | float           | newsguard score                                                        |     |

## Linguistic Features and How to Extract Them

**What Mathias already did**:

1. Toxicity Levels: Google Perspective API
2. Action Quotient: TIGER corpus POS tagging
	- ratio of verbs to adjectives
3. Media Quality: Newsguard
4. Engagement Rate: 
	- Content: average monthly bundle length =  messages per user each month
	- Participation: messages by author to group
	- Replication: share of messages that were directly replied to

```New Possibilities:```
```- gatekeeping```
```- emotive language (maybe TextBlob or VADER?)/ aggresiveness```
```- measure for shared vs. own content```
```- ratios for how much content deals with each of the topics```
```- (im)politeness, humor, indexicality, and multilingualism (https://journals.sagepub.com/doi/full/10.1177/21582440211047572) ```

**(Supervised ML or Dictionary Based) Numeric and POS features**
https://link-springer-com.ep.fjernadgang.kb.dk/article/10.1007/s11042-023-15216-0
- e.g. positive words, negative words, nouns, adjectives etc
https://ieeexplore-ieee-org.ep.fjernadgang.kb.dk/document/9940948
- first person, second person pronouns, superlatives, swear words, sentiment & emotion score etc (dictionary based)
- contextual features: number of replies
https://www-sciencedirect-com.ep.fjernadgang.kb.dk/science/article/pii/S0360835222004697?via%3Dihub
- number of special characters, upper & lowercase, long sentences
- stylometric features: number of articles, determinant count, noun count, verb count, adverb count, number of syllables, number of words, no of total sentence, rate of the noun, adjective count, and rate of adverb

_--> POS Tagging via TIGER CORPUS


**Readability/ Complexity**
https://www-sciencedirect-com.ep.fjernadgang.kb.dk/science/article/pii/S0360835222004697?via%3Dihub
- different scores: Gunning Fog grade Index, Coleman Liau Index, Linsear Write, Dale-Chall Readability, Flesch Readability Index, Spache Readability Index, Automatic readability Index
- based on stylometric features (see above)

 _-->TextStat can compute Flesch Readability in German_ (https://formative.jmir.org/2022/8/e35563)
 _--> https://huggingface.co/krupper/text-complexity-classification four complexity classes_
  ~~--> https://klartext.uni-hohenheim.de/hix Online Tool for German text~~
  _hugging face: BERT based https://huggingface.co/MiriUll/distilbert-german-text-complexity

**Sentiment Analysis/ Political Polarity**
https://link-springer-com.ep.fjernadgang.kb.dk/article/10.1007/s41060-023-00469-7
- using SVM (needs labelled data)
https://www-sciencedirect-com.ep.fjernadgang.kb.dk/science/article/pii/S0360835222004697?via%3Dihub
- polarity: positive/ negative

~~_GerVADER NOT an option --> BAD results;_~~
_--> germansentiment looks promising https://huggingface.co/oliverguhr/german-sentiment-bert
OR https://huggingface.co/aari1995/German_Sentiment?text=yayy+niemand+freut+sich_


**lexical features**
https://www-sciencedirect-com.ep.fjernadgang.kb.dk/science/article/pii/S0306457324000517?via%3Dihub
- lexical density: higher density = more effective information conveyed --> done by counting + calculating ratio of content vs. function words (how??)
- lexical sophistication: proportion of uncommon words in the text and the complexity of lexical representation --> Tool for the Automatic Analysis of Lexical Sophistication (TAALES) (english only)
- syntactic complexity

**Narrative Novelty vs. Continuity**
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0293508
- cosine distance of vector representations of messages (or 'concepts' in messages)
- novelty: avg distance to preceding messages
- transience: avg distance to following messages
- resonance = diff between novelty and transience
- six categories based on these three scores:
	- isolated
	- continued
	- emerging
	- fading
	- continued-emerging
	- continued-fading
- (findings show that channels usually have a lot of isolated messages)

https://cris.vub.be/ws/portalfiles/portal/94099959/short5.pdf
- keyness analysis to determine new emerging topics

**~~Linguistic Inquiry Word Count (LIWC)**
- ~~application~~
- ~~gives ratios of word in corpus e.g. positive, negative, allure etc.~~
- ~~hard to see what it can do~~


to check:
- [x] Web of Science
- [ ] WOS: linguistic + specific conspiracy name
- [x] websearch for German NLP tools
- [x] hugging face
- [x] GPT


# Tuesday, 25/06/2024

>[!TASKS]
>- [x] running task: research into linguistic features
>- [x] end of internship talk: Irene (migration)

# Wednesday, 26/06/2024

>[!TASKS]
>- [x] running task: research into linguistic features
>- [x] Workshop: Deep Roots of Political and Economic Development
>- [x] Wednesday Talk: Collapse vs. Stability of Complex Systems

# Thursday, 27/06/2024

>[!TASKS]
>- [x] meeting w/ Henrik & Mathias about linguistic features
>- [x] Workshop: Deep Roots of Political and Economic Development
>  
>**UPDATE AFTER MEETING:**
>  - [x] add notes above to Google Doc
>        
>**New Running Tasks**
>  - [ ] look into emoji classifier/ emoji use for sarcasm detection
>  - [ ] look into definitions of complexity classifier
>  - [ ] look into German LWIC: how to use, license needed etc
>  - [ ] get estimate for runtime (on ~200.000 rows)
>  - [ ] enrich dataset with:
> 	 - [ ] count-based features
> 	 - [ ] POS tagging
> 	 - [ ] emoji classifier
> 	 - [ ] language complexity
> 		 - [ ] hugging face model
> 		 - [ ] textstat Flesch readability ease score
> 	 - [ ] sentiment: German LIWC
> 	 - [ ] narrative novelty vs. continuity

## Definitions for Language Complexity

https://huggingface.co/krupper/text-complexity-classification

--> no real documentation/ accompanying paper available. so probably not suitable for our purposes

**ALTERNATIVE**:

HuggingFace Model: https://huggingface.co/MiriUll/distilbert-german-text-complexity
Accompanying Paper: https://aclanthology.org/2022.germeval-1.4.pdf

- score between 1 and 7, where higher score indicates higher complexity
- script available on git


## Emoji Classifier

- https://www-degruyter-com.ep.fjernadgang.kb.dk/document/doi/10.1515/ip-2023-5001/html paper indicates that emojis incongruent to message indicate irony/sarcasm

- EmojiNet API usable to extract meaning of emojis --> calls time out
- 

# Friday, 28/06/2024
>[!TASKS]
>- [x] Workshop: Deep Roots of Political and Economic Development
> - [ ] running task: write/test code to enrich dataset with 
> 	- [x] count-based features
> 	- [x] POS tagging
> 	- [x] Flesch Reading Ease

## Count Based Features
features relate to either message or fwd_message, depending on which of the two is available

| Feature           | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| sent_count        | num of sentences                                                         |
| word_count        | num of words                                                             |
| avg_sent_length   | average sentence length                                                  |
| avg_word_length   | average word length                                                      |
| exclamation_count | count of exclamations, multiple consecutive ! counted as one exclamation |
| question_count    | count of questions, multiple consecutive ? counted as one question       |
| emoji_count       | count of emojis                                                          |

## POS Tags

done with Spacy instead of TIGER Corpus, cause Spacy is trained on TIGER Corpus and would be needed to be loaded for preprocessing anyways

| Feature    | Description          |     |
| ---------- | -------------------- | --- |
| noun_count | number of nouns      |     |
| verb_count | number of verbs      |     |
| adj_count  | number of adjectives |     |

## Flesch Reading Ease

| Feature             | Description               |
| ------------------- | ------------------------- |
| flesch_reading_ease | Flesch Reading Ease Score |

