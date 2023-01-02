import tweepy
import credentials as cd
import pymongo

# instantiate a tweepy client
client_tweepy = tweepy.Client(
    bearer_token=cd.BEARER_TOKEN,
    wait_on_rate_limit=True,
)

#get the user for BernieSanders
response = client_tweepy.get_user(
    username='BernieSanders',
    user_fields=['created_at', 'description', 'location',
                 'public_metrics', 'profile_image_url']
)
user = response.data

# get client connection to mongodb container
client_mongo = pymongo.MongoClient(host=cd.HOST, port=cd.PORT)

# empty the database
client_mongo.drop_database('db_tweets')

# create db 
db = client_mongo.db_tweets # 'db_tweets' is name of db

# create collection
dbcoll = db.collection_tweets 

# request user tweets and return cursor
cursor = tweepy.Paginator(
    method=client_tweepy.get_users_tweets,
    id=user.id,
    exclude=['replies', 'retweets'],
    tweet_fields=['author_id', 'created_at', 'public_metrics']
).flatten(limit=100)

# insert tweets into mongoDB container
for tweet in cursor:
    dbcoll.insert_one(dict(tweet))
