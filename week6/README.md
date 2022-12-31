### datapipeline with docker containers (using a yml-file)

#### target:
download tweets from a chosen user. Perform a sentiment analysis on them and upload the results to slack with a slack bot. 

#### Following 5 containers will be used and tasks will be performed:
- mongo_cont: container which contains a mongodb database
- tweepy container (runs script get_tweets_mdb.py): 
    gets latest 100 tweets from Bernie Sanders' Twitter account using tweepy API
    and stores them in the mongoDB database within the mongodb container (mongo_cont).
- postgres_cont: container containing a postgres-database
- sa_tweets (runs script sa_tweets_2.py): ETL
    downloads the tweets from the mongoDB database. Cleans the tweets and performs
    a sentiment analysis on the tweets with the SentimentIntensityAnalyzer from VaderSentiment.
    the tweets with their compound sentiment score will be uploaded into the postgres database on 
    the postgres container (postgres_cont)
- slackbot_cont (runs script slackbot.py):
    queries the tweets and the compound score from the postgres database and
    posts them on slack