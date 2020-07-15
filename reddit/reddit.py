import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import praw
import pdb

sia = SIA()

# Create reddit instance
reddit = praw.Reddit(client_id='1sbu376RCBiWRw',
                     client_secret='NbqiHMPiKicBXvgfrID-xVNktZM',
                     user_agent='mac:cc_framing:v1 (by /u/emma_cc_research)')

subreddit = reddit.subreddit("redditdev")
print(subreddit.display_name)  # Output: redditdev
print(subreddit.title)         # Output: reddit Development
print(subreddit.description)   # Output: A subreddit for discussion of ...

headlines = set()
df = pd.DataFrame( columns=['Author','upvotes','upvote_ratio','title', 'body',\
	'pos_sentiment_score','neg_sentiment_score','neutral_sentiment_score'])

sr_name = 'climatechange'


for submission in reddit.subreddit(sr_name).top(limit=None):
	author = submission.author
	upvotes = submission.score
	upvote_ratio = submission.upvote_ratio
	title = submission.title
	body = submission.selftext
	title_polarity = sia.polarity_scores(title)
	sen_pos = title_polarity['pos']
	sen_neg = title_polarity['neg']
	sen_neu = title_polarity['neu']
	df.loc[len(df),:] = [author, upvotes, upvote_ratio, title, body,\
		sen_pos, sen_neg, sen_neu]

	print('%s -> %s\n' % (title_polarity, title))


df.to_csv('subreddit_%s.csv' % sr_name)
