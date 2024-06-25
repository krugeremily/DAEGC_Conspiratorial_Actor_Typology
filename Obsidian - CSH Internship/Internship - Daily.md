https://docs.google.com/document/d/1B9S35aB3A0MxObnLSVdA7QrruzmPaArz_OkyyoquPr8/edit?pli=1
# Monday, 24/06/2024

**Tasks:**

- Find further linguistic features and a way to extract them
- Understand the dataset and each column

## Understanding Dataset Columns

**Dataset: selected_groups_with_transcriptions.csv.gzip**

| Index | Column                   | Type            | Content                                                                |
| ----- | ------------------------ | --------------- | ---------------------------------------------------------------------- |
| 0     | UID_key                  |                 | unique ID: hashed group name + message ID                              |
| 1     | initial_ID               | int             |                                                                        |
| 2     | mid_message              | float           | ID of message in respective group                                      |
| 3     | mid_file                 | float           | ID of media file in respective group                                   |
| 4     | group_name               | string + emojis | name of the group/channel                                              |
| 5     | posting_date             |                 | date and time of message in the group                                  |
| 6     | message                  | string + emojis | message content                                                        |
| 7     | fwd_message              | float           | forwarded message content                                              |
| 8     | fwd_posting_date_message | float           | for forwarded messages: date and time of message in the original group |
| 9     | posting_date_file        |                 | for files: date and time of file in the group                          |
| 10    | link_url                 |                 | url as found in HTML a-tag                                             |
| 11    | media_file               |                 | name of media file                                                     |
| 12    | media_file_type          |                 | voice_message, video, or photo                                         |
| 13    | fwd_posting_date_file    |                 | for forwarded files: date and time of file in the original group       |
| 14    | fwd_link_url             |                 | for forwarded messages: url as found in HTML a-tag in original message |
| 15    | fwd_media_file           |                 | for forwarded messages: name of forwarded media file                   |
| 16    | fwd_media_file_type      |                 | for forwarded messages: voice_message, video, or photo                 |
| 17    | author                   | float           | hashed name for author of message                                      |
| 18    | fwd_author               | float           | for forwarded messages: hashed name for author of original message     |
| 19    | day                      | float           | day of a month (1-31)                                                  |
| 20    | week                     | float           | week of year (1-52)                                                    |
| 21    | weekday                  | float           | day of week (1-7)                                                      |
| 22    | message_hash             | float           | hashed message content                                                 |
| 23    | fwd_message_hash         | float           | for forwarded messages: hashed message content of forwarded message    |
| 24    | website                  |                 | website domain, if link_url refers to a website                        |
| 25    | replied_to               |                 | message_id of message that has been replied to                         |
| 26    | year                     | float           | year                                                                   |
| 27    | month                    | float           | month of year (1-12)                                                   |
| 28    | day_of_year              | float           | day of a year (1-366)                                                  |
| 29    | duration                 | float           | duration of media file (in seconds?)                                   |
| 30    | filepath                 |                 | filepath to media file                                                 |
| 31    | filename                 |                 | same as media_file                                                     |
| 32    | filename_if_joined       |                 | for joining transcription dataset                                      |
| 33    | transcribed_message      |                 | transcription of media file                                            |
| 34    | newsguard_domain         |                 | newsguard webiste domain                                               |
| 35    | newsguard_score          | float           | newsguard score                                                        |

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

**New Possibilities:**

- gatekeeping
- emotive language (maybe TextBlob or VADER?)
- measure for shared vs. own content
- ratios for how much content deals with each of the topics
- (im)politeness, humor, indexicality, and multilingualism (https://journals.sagepub.com/doi/full/10.1177/21582440211047572)

to check:
- Web of Science
	- linguistic + specific conspiracy name
- hugging face
- GPT
