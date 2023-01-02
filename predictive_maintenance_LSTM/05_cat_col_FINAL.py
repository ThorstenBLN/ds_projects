import pandas as pd
import numpy as np
import datetime
import os.path
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

!pip install keras

from tensorflow import keras, device
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras import backend as K
from keras.metrics import Precision, Recall

from google.colab import files
import io

# definition of data parameters
DAYS_MEAN = 6 # optimized !! [6, 9, 12]
DAYS_VAR = 12 # optimized !![6, 9, 12]
SEQUENCE_LENGTH = 50 # must be !! defined by regression model

# definition of model parameters
N_EPOCHS = 15
N_BATCH_SIZE = 64 # optimized !! [32, 48, 64]
LSTM_1_UNITS = 32 # optimized !! [32, 48, 64]
LSTM_2_UNITS = 24 # optimized !! [24, 32, 40]
LSTM_3_UNITS = 24 # optimized !! [8, 16, 24]

# upload data files to google collab
uploaded = files.upload()
df_train = pd.read_csv(io.BytesIO(uploaded['train_cat_set1.csv']), header=0)
df_test = pd.read_csv(io.BytesIO(uploaded['test_cat_set1.csv']), header=0)

filename = 'hyperparams_classification_FINAL.csv'
path = './'
header = ['timestamp','sequence_length','days_mean','days_var','N_EPOCHS','N_BATCH_SIZE','LSTM_1_UNITS','LSTM_2_UNITS','LSTM_3_UNITS','Test_loss','Test accuracy','Test precision','Test recall','cm_row0','cm_row1','cm_row2', 'FP1', 'FN_1', 'FP2', 'FN2']
# create log file and add header if it doesn't exist yet
if not os.path.exists(os.path.join(path, filename)):
    print(os.path.join(path, filename))
    file = open(os.path.join(path,filename), mode='w')
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    file.close()

# open the log-file for the loop
file = open(os.path.join(path,filename), mode='a')
csv_writer = csv.writer(file)


def add_mean_var(df, days_mean=5, days_var=5):
    '''adding the rolling mean and rolling variance for all sensors of the input dataframe. 
    The length of the rolling sequence is defined by seperately by the 2 input parameters'''    
    # create a df with the mean of the parameter-days with the same index as the original df
    no_days_mean = days_mean
    names = ['mean_' + feature for feature in FEATURES]
    df_mean = df.groupby('id')[FEATURES].rolling(no_days_mean).mean()
    df_mean.columns = names
    df_mean.reset_index(inplace=True)
    df_mean.drop(columns=['level_1', 'id'], inplace=True)
    
    # create a df with the var of the parameter-days with the same index as the original df
    no_days_var = days_var
    names = ['var_' + feature for feature in FEATURES]
    df_var = df.groupby('id')[FEATURES].rolling(no_days_var).var()
    df_var.columns = names
    df_var.reset_index(inplace=True)
    df_var.drop(columns=['level_1', 'id'], inplace=True)

    return df_mean, df_var

def create_sequences(df_eng, seq_len, features, labels):
    '''create the x and y data sequences for a single engine. The features of x data is defined by 
    the input list features, the labels for the y datais defined by the input list labels.
    Length of the sequence is defined by seq_len'''  
    X_samples = []
    y_samples = []
    for i in range(df_eng.shape[0]):
        if i + seq_len - 1 >= df_eng.shape[0]:
            break
        X_samples.append(df_eng[features].iloc[i : i + seq_len]) # appending a list of all feature elements to the list
        y_samples.extend(df_eng[labels].iloc[i + seq_len - 1]) # just add the y-value to a list (1D) - appending would create 2D
    print(df_eng['id'].unique(), np.array(X_samples).shape, np.array(y_samples).shape)
    return np.array(X_samples), np.array(y_samples)

def create_input(df, sequence_length, features, labels):
    '''creates X an y input data for the lstm. Iterates over all engine ids, creates sequences and stacks
    them to the final LSTM-ready data'''
    # iterate over all engine id's to create the samples and concattenate them
    X = []
    y = []
    for engine_id in df['id'].unique():
        # check if the dataframe is long enough for a sample
        if df.loc[df['id'] == engine_id].shape[0] < sequence_length:
            continue
        # else create the X- and y-samples and append them to the list
        X_sample, y_sample = create_sequences(df.loc[df['id'] == engine_id], sequence_length, features, labels) 
        X.append(X_sample)
        y.append(y_sample)
    # bring the arrays into the LSTM-Format
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# reads the data files into the train and test dataframes
df_train = pd.read_csv('train_cat_set1.csv', header=0)
df_test = pd.read_csv('test_cat_set1.csv', header=0)
df_train.drop(columns=['Unnamed: 0'], inplace=True)
df_test.drop(columns=['Unnamed: 0'], inplace=True)

# create min max scaler for features
INFO = ['id', 'cycle', 'rul', 'label']
FEATURES = df_train.drop(columns=INFO).columns
scaler = MinMaxScaler()
df_feat = pd.DataFrame(data=scaler.fit_transform(df_train[FEATURES]), columns=scaler.get_feature_names_out(), index=df_train.index)
df_train = pd.merge(left=df_train[INFO], right=df_feat, left_index=True, right_index=True)
df_feat_test = pd.DataFrame(data=scaler.transform(df_test[FEATURES]), columns=scaler.get_feature_names_out(), index=df_test.index)
df_test = pd.merge(left=df_test[INFO], right=df_feat_test, left_index=True, right_index=True)

# add the means and vars to train and test data
df_train_mean, df_train_var = add_mean_var(df_train, DAYS_MEAN, DAYS_VAR)
df_test_mean, df_test_var = add_mean_var(df_test, DAYS_MEAN, DAYS_VAR)

# merge the original dataframe with the mean and variance dataframes
df_train = df_train.merge(df_train_mean, left_index=True, right_index=True)
df_train = df_train.merge(df_train_var, left_index=True, right_index=True)
df_test = df_test.merge(df_test_mean, left_index=True, right_index=True)
df_test = df_test.merge(df_test_var, left_index=True, right_index=True)

# drop the na which have occured due to the rolling mean/variance
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# redefine the features as there have been added the mean/var
FEATURES = df_train.drop(columns=INFO).columns
labels = ['label']

# create the input arrays for train and test
X_train, y_train_int = create_input(df_train, SEQUENCE_LENGTH, FEATURES, labels)
y_train_cat = to_categorical(y_train_int, 3)
print(X_train.shape, y_train_int.shape, y_train_cat.shape)

X_test, y_test_int = create_input(df_test, SEQUENCE_LENGTH, FEATURES, labels)
y_test_cat = to_categorical(y_test_int, 3)
print(X_test.shape, y_test_int.shape, y_test_cat.shape)

# create and train the LSTM model
with device('/device:GPU:0'):
    K.clear_session()
    lstm_model = Sequential(
        [   # 1st layer (LSTM layer)
            keras.layers.LSTM(
                input_shape=(SEQUENCE_LENGTH, len(FEATURES)),
                units=LSTM_1_UNITS,
                return_sequences= True # gives back the hidden state for each timestep (needed for the input of LSTM-layer 2)
                ),
            keras.layers.Dropout(0.4),
            # 2nd layer (LSTM layer)
            keras.layers.LSTM(units=LSTM_2_UNITS,
                return_sequences= True
                ),
            keras.layers.Dropout(0.4),
            keras.layers.LSTM(units=LSTM_3_UNITS
                ),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(units=3, activation='softmax')
        ]
    )
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
    lstm_model.summary()

    # fit the LSTM
    history = lstm_model.fit(X_train, y_train_cat, 
                epochs=N_EPOCHS, 
                batch_size=N_BATCH_SIZE, 
                validation_split=0.1,
                class_weight={0: 1, 1: 2, 2: 2}, # to balance the unbalanced data (put more weight on class 1) 
                verbose=1, # defines if animation will be shown while training
                #callbacks = [EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')]
                )

    # evaluate the model and print the results
    score = lstm_model.evaluate(X_test, y_test_cat, verbose=0)
    print(score)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Test precision:", score[2])
    print("Test recall:", score[3])

    y_pred_train = lstm_model.predict(X_train)
    y_pred_test = lstm_model.predict(X_test)

    y_train_int = np.argmax(y_train_cat, axis=1)
    y_test_int = np.argmax(y_test_cat, axis=1)
    y_pred_train_int= np.argmax(y_pred_train, axis=1)
    y_pred_test_int= np.argmax(y_pred_test, axis=1)

    cm = confusion_matrix(y_test_int,y_pred_test_int)
    print(cm)
# write results to hyperparam csv
results = [datetime.datetime.now(),SEQUENCE_LENGTH,DAYS_MEAN,DAYS_VAR,N_EPOCHS,N_BATCH_SIZE,LSTM_1_UNITS,LSTM_2_UNITS,LSTM_3_UNITS,score[0],score[1],score[2],score[3],cm[0],cm[1],cm[2], cm[0][1], cm[1][0], cm[1][2], cm[2][1]]
csv_writer.writerow(results)

file.close()
files.download('hyperparams_classification_FINAL.csv')
# save trained model
lstm_model.save('class_model.h5')