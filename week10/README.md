### recommender system for movies

#### target:
create a simple Flask webpage containing a movie recommender system using non negative matrix factorization.  
For proof of concept purpose create also a movie recommender with cosine similarity.

#### 5 files:
- 10.2_movie_db.ipnyb: 
    simple movie recommender (with collaborative filtering). 
    - It just recommends k movies with the best average evaluation which the user hasn't seen yet.
- 10.3_movie_db_nmf.ipynb:
    movie recommender with non negative matrix factorization
    - fit the NMF model with know evaluations of all users 
    - calculate the P-matrix of the new user using the NMF-model
    - calculate a new R-hat for the user and recommend the k best movies
- 10.4_movie_db_content_based.ipynb:
    movie recommender with cosine similarity (proof of concept)
- movies.py & recommender.py:
    a movie recommender webpage with Flask. (in 90s style - on purpose)
    - first prompts the user to evaluate 5 movies
    - based on that creates movie recommendations for the user with the saved NMF-model