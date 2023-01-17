### text classification model on song lyrics

#### target: 
predict the artist from a piece of text

#### files:
- 4_4_lyrics_pipe: this script downloads song lyrics from certain artists and saves each song in a file in the corresponding artist folder. It also contains 2 simple models to make a first prediction test (using tfidf-vectorizer with a NaiveBayesClassifier or RandomForestClassifier).
- 4_6_doubled_filed: as some songs might have multiple publications  this script eliminates these duplicates (e.g. live or remastered versions)
- 4_5_lyrics_pipeline: this script downloads all lyrics into a pandas dataframe and predicts the corresponding artist with A. a RandomForestClassifier and B. a NaiveBayesClassifier. For both models a TF-IDF-Vectorizer and a RandomOverSampler (due to different number of songs per artist) have been used. On both pipelines a hyperparameter optimization has been performed. The best 2 models have been saved.
- lyrics.py: in this commandline script the user can enter a songtext and both models will predict the corresponding artist