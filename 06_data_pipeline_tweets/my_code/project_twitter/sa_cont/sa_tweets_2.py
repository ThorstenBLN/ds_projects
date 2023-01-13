import pymongo
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
import credentials as cd
import psycopg2
import logging
from sqlalchemy import create_engine
import time


def clean_tweets(tweet):
    '''removes the defined patterns from the tweet input parameter (subsitute with '').
    returns the cleaned tweet'''
    # patterns to be removed
    mentions_regex= '@[A-Za-z0-9]+'
    url_regex='https?:\/\/\S+' #this will not catch all possible URLs
    hashtag_regex= '#'
    rt_regex= 'RT\s'
    # remove the patterns from the tweet
    tweet = re.sub(mentions_regex, '', tweet)  #removes @mentions
    tweet = re.sub(hashtag_regex, '', tweet) #removes hashtag symbol
    tweet = re.sub(rt_regex, '', tweet) #removes RT to announce retweet
    tweet = re.sub(url_regex, '', tweet) #removes most URLs
    return tweet

# sleep 10 seconds to be sure that mongodb is created and filled with tweets
time.sleep(10)

# download the tweets from mongoDb into a cursor
client_mongo = pymongo.MongoClient(host="mongo_cont", port=27017)
dbcoll = client_mongo.db_tweets.collection_tweets
cursor_mongo = dbcoll.find({}, {'id': 1, 'text': 1})

# create a 2D list with [tweet-id from mongoDB, cleaned tweet]
cleaned_tweets = [[document['id'], clean_tweets(document['text'])] for document in cursor_mongo]
print(f"{len(cleaned_tweets)} tweets downloaded from mongoDB")

# upload the cleaned tweets into a DF
df_tweets = pd.DataFrame(cleaned_tweets, columns=['id', 'cleaned_text'])

# perform sentiment analysis on df
s_analyzer = SentimentIntensityAnalyzer()
df_score = df_tweets['cleaned_text'].apply(s_analyzer.polarity_scores).apply(pd.Series)
df_tweets = df_tweets.merge(df_score, left_index=True, right_index=True)
print(f"size tweets dataframe: {df_tweets.shape}")

# connect to database create table tweets
conn_string = f'postgresql://{cd.USERNAME}:{cd.PASSWORD}@{cd.HOST}:{cd.PORT}/{cd.DATABASE_NAME}'
client_psql = create_engine(conn_string,echo=False)
client_psql_con = client_psql.connect()
df_tweets = pd.DataFrame(df_tweets)
no_rows = df_tweets.to_sql(cd.TABLE_NAME, con=client_psql_con, if_exists='replace',
          index=False)
print(f"POSTGRESQL: {no_rows} inserted into {cd.TABLE_NAME}")
