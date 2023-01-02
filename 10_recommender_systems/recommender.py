import pandas as pd
from sklearn.decomposition import NMF 
import pickle
import numpy as np
import random

def get_recommendation(user_query, model, k, user_item_f):
    # get movie names out of modes
    movies = model.feature_names_in_
    # get the Q-Matrix
    comp = ["nmf_" + str(i + 1) for i in range(model.n_components_)]
    Q_df = pd.DataFrame(data=model.components_,
        columns=model.feature_names_in_,
        index=comp)
    # create a user dataframe
    df_user = pd.DataFrame(data=user_query, index=['new_user'], columns=movies)
    df_user_imp = df_user.fillna(user_item_f.mean())
    # create the P matrix for the user
    P_user = model.transform(df_user_imp)
    R_hat_user = pd.DataFrame(data=np.dot(P_user, Q_df), columns=movies, index=['user'])
    result_df = R_hat_user.T.sort_values(by='user', ascending=False)
    return result_df.head(k)


