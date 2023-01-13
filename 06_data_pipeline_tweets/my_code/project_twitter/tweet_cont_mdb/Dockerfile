FROM python:3.8

WORKDIR /tweepy

COPY . /tweepy

RUN pip install --trusted-host pypi.python.org -r requirements_twitter.txt

CMD ["python", "get_tweets_mdb.py"]