# image file (python environment)
FROM python:3.8
# working directory in the container
WORKDIR /sa
# copy files from current local host folder to the container folder /sa
COPY . /sa
# runs command on commandline to pip install all packages from the requirement
# (RUN is an image build step, the state of the container after a RUN command 
# will be committed to the container image.)
RUN pip install --trusted-host pypi.python.org -r requirements_sa.txt
#  CMD is the command the container executes by default when you launch the built image
CMD ["python", "sa_tweets_2.py"]