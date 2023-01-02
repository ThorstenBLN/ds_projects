from sqlalchemy import create_engine
import credentials as cd
import pandas as pd
import json
import requests
import psycopg2
import time

# wait for the sentiment analysis to be done and saved in postgeSQL database
time.sleep(20)
# connect to database
conn_string = f'postgresql://{cd.USERNAME}:{cd.PASSWORD}@{cd.HOST}:{cd.PORT}/{cd.DATABASE_NAME}'
client_psql = create_engine(conn_string,echo=False)
client_psql.connect()
# extract 5 entries from postgreSQL
extract_string = f"SELECT * FROM {cd.TABLE_NAME} LIMIT 5;"
rs_tweets = client_psql.execute(extract_string).fetchall()
# post all 5 tweets and it's compound score to slack
df = pd.DataFrame(rs_tweets)
print(f"SLACKBOT: {df.columns}")
for message, mood in zip(df['cleaned_text'], df['compound']):
    mood_str = "neutral"
    if float(mood) > 0.2:
        mood_str = "happy"
    elif float(mood) < -0.2:
        mood_str = "grumpy"
    message = f"""Bearnie Sanders is in a {mood_str} mood ({mood}) and sends you this message:\n
                {message}."""
    json_message = {'text': message}
    requests.post(url=cd.WEBHOOKURL, json=json_message)


    

