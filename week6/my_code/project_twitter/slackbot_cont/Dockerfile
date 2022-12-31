FROM python:3.8
WORKDIR /bot
COPY . /bot
RUN pip install --trusted-host pypi.python.org -r requirements_bt.txt
CMD ["python", "slackbot.py"]