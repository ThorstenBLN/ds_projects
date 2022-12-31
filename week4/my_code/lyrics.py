import pickle
from argparse import ArgumentParser

if __name__ == "__main__":
    # loading both models
    with open("model_RFC.pickle","rb") as file:
        model_RFC = pickle.load(file)

    with open("model_MNB.pickle","rb") as file:
        model_MNB = pickle.load(file)

    # user has to run the script and add a sontext as argument
    parser = ArgumentParser(description="enter: python lyrics.py songtext")
    parser.add_argument("lyrics",
                        help= "please enter lyrics from a song")
    # predict the artist to the lyricsinput of the user with both models
    user_input = parser.parse_args()
    song_input = user_input.lyrics
    artist_RFC = model_RFC.predict([song_input.lower()])
    artist_MNB = model_MNB.predict([song_input.lower()])
    print(f"\nRFC says the song is from: {artist_RFC[0]}\nMNB says the song is from: {artist_MNB[0]}")

