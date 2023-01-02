import pandas as pd
import numpy as np
import os.path 
import csv
import datetime

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

!pip install keras
!pip install tensorflow_addons

from tensorflow import keras, initializers, device
# from keras.utils import to_categorical
from keras.models import Sequential
from keras import backend as K
import tensorflow_addons as tfa

from google.colab import files
import io

DATASET = 2 # 1: original datafiles 1 & 3 // 2: original datafiles 2 & 4
LOG_Y = True # optimum: (DATASET 1: False) (DATASET 2: XXX)
SEQUENCE_MEAN = False #create the mean of the whole sequence for all features
SAVE_MODEL = False # save the model after training

# defining the parameters of the LSTM-model
N_EPOCHS_LIST = [40] # [60, 50, 40] # optimum: (DATASET 1: 40) (DATASET 2: XXX)
N_BATCH_SIZE_LIST = [48] # [64, 96, 128] # optimum: (DATASET 1: 40) (DATASET 2: XXX)
LSTM_1_UNITS_LIST = [64] # opt [32, 40, 48] # optimum: (DATASET 1: 32) (DATASET 2: 64)
LSTM_2_UNITS_LIST = [24] # opt [16, 24, 32] # optimum: (DATASET 1: 32) (DATASET 2: 24)
DENSE_1_UNITS_LIST = [48] # opt [24, 32, 40] # optimum: (DATASET 1: 24) (DATASET 2: 48)
LSTM_DROPOUT_LIST = [0.3] #[0.2, 0.3, 0.4] # optimum: (DATASET 1: 0.3) (DATASET 2: XXX)
DENSE_DROPOUT_LIST = [0.3] #[0.2, 0.3, 0.4] # optimum: (DATASET 1: 0.3) (DATASET 2: XXX)
VALIDATION_SPLIT_LIST = [0.1]

# defining the parameters of the data
SEQUENCE_LENGTH_LIST = [45] # [30, 40, 50] # optimum: (DATASET 1: 50) (DATASET 2: XXX)
Y_THRESHOLD_LIST = [75] # [75, 80, 85, 90, 95] # optimum: (DATASET 1: 70) (DATASET 2: XXX) # cutting of the regression to higher rul
DAYS_MEAN_LIST = [3, 6, 9] # [6, 8, 10, 12] # optimum: (DATASET 1: 6) (DATASET 2: XXX) # params of rolling and vars to train and test data
DAYS_VAR_LIST = [9, 12, 15] # [6, 8, 10, 12] # optimum: (DATASET 1: 12) (DATASET 2: XXX)

# Thresholds for RMSE-Metrics
THRESHOLD_0_VAL = 60
THRESHOLD_1_VAL = 30
THRESHOLD_2_VAL = 15

def add_mean_var(df, days_mean=5, days_var=5):
    '''adding the rolling mean and rolling variance for all sensors of the input dataframe. 
    The length of the rolling sequence is defined by seperately by the 2 input parameters'''  
    # create a df with the mean of the parameter-days with the same index as the original df
    no_days_mean = days_mean
    names = ['mean_' + feature for feature in FEATURES_ORIG]
    df_mean = df.groupby('id')[FEATURES_ORIG].rolling(no_days_mean).mean()
    df_mean.columns = names
    df_mean.reset_index(inplace=True)
    df_mean.drop(columns=['level_1', 'id'], inplace=True)
    
    # create a df with the var of the parameter-days with the same index as the original df
    no_days_var = days_var
    names = ['var_' + feature for feature in FEATURES_ORIG]
    df_var = df.groupby('id')[FEATURES_ORIG].rolling(no_days_var).var()
    df_var.columns = names
    df_var.reset_index(inplace=True)
    df_var.drop(columns=['level_1', 'id'], inplace=True)
    return df_mean, df_var

def create_tot_mean(df, names, features):
    '''creates the mean for the whole sequence.'''
    df[names] = df[features].mean()
    return df

def create_sequences_2(df_eng, seq_len, labels):
    '''create the x and y data sequences for a single engine. Additionally to create_sequences it adds
    the mean of the whole sequence for each feature. The features of x data is defined by 
    the input list features, the labels for the y datais defined by the input list labels.
    Length of the sequence is defined by seq_len'''
    X_samples = []
    y_samples = []
    # define the features and names for the FE
    FEATURES_MEAN = FEATURES_ORIG
    names_mean = ["mean_tot_" + str(feature) for feature in FEATURES_MEAN]
    for i in range(df_eng.shape[0] - seq_len + 1):
        # create a sequence-df with the total mean added
        df = create_tot_mean(df_eng.iloc[i : i + seq_len].copy(), names_mean, FEATURES_MEAN)
        X_samples.append(df.drop(columns=INFO)) # appending a list of all feature elements to the list
        y_samples.extend(df[labels].iloc[-1]) # just add the last y-value to a list (1D) - appending would create 2D
    print(df_eng['id'].unique(), np.array(X_samples).shape, np.array(y_samples).shape)
    return np.array(X_samples), np.array(y_samples)


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
    return np.array(X_samples), np.array(y_samples)


def create_input(df, sequence_length, features, labels):
    '''creates X an y input data for the lstm. Iterates over all engine ids, creates sequences and stacks
    them to the final LSTM-ready data'''
    X, y = [], []
    for engine_id in df['id'].unique():
        # check if the dataframe is long enough for a sample
        if df.loc[df['id'] == engine_id].shape[0] < sequence_length:
            continue
        # else create the X- and y-samples and append them to the list
        if SEQUENCE_MEAN:
            X_sample, y_sample = create_sequences_2(df.loc[df['id'] == engine_id], sequence_length, labels)     
        else:
            X_sample, y_sample = create_sequences(df.loc[df['id'] == engine_id], sequence_length, features, labels) 
        X.append(X_sample)
        y.append(y_sample)
    # bring the arrays into the LSTM-Format
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# define filenames for the 2 different datasets
if DATASET == 2:
    data_file_train = 'train_cat_set3.csv'
    data_file_test = 'test_cat_set3.csv'
    params_file = 'hyperparams_regression_set2.csv'
else:
    data_file_train = 'train_cat_set1.csv'
    data_file_test = 'test_cat_set1.csv'
    params_file = 'hyperparams_regression.csv'

# create log file and add header if it doesn't exist yet
path = './'
header = ['timestamp','y_threshold','sequence_length','days_mean','days_var','N_EPOCHS','N_BATCH_SIZE','LSTM_1_UNITS','LSTM_2_UNITS','DENSE_1_UNITS','LSTM_DROPOUT','DENSE_DROPOUT','VALIDATION_SPLIT','RSME_train','R2_train','RSME_val','R2_val','RMSE_tot','R2_tot','RMSE_60','mean_delta_60','std_delta_60','outlier_max_60','outlier_min_60','RMSE_30','mean_delta_30','std_delta_30','outlier_max_30','outlier_min_30','RMSE_15','mean_delta_15','std_delta_15','outlier_max_15','outlier_min_15']
if not os.path.exists(os.path.join(path, params_file)):
    print(os.path.join(path, params_file))
    file = open(os.path.join(path,params_file), mode='w')
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    file.close()

# open the log-file for the loop
file = open(os.path.join(path,params_file), mode='a')
csv_writer = csv.writer(file)

# upload data files to google collab
uploaded = files.upload()
df_train = pd.read_csv(io.BytesIO(uploaded[data_file_train]), header=0)
df_test = pd.read_csv(io.BytesIO(uploaded[data_file_test]), header=0)

# iterate over data-relevant parameters for hyper-params-optimization
for SEQUENCE_LENGTH in SEQUENCE_LENGTH_LIST:
    for Y_THRESHOLD in Y_THRESHOLD_LIST:
        for DAYS_MEAN in DAYS_MEAN_LIST:
            for DAYS_VAR in DAYS_VAR_LIST:
                # read the files if they are already uploaded to colab
                df_train = pd.read_csv(data_file_train, header=0)
                df_test = pd.read_csv(data_file_test, header=0)
                df_train.drop(columns=['Unnamed: 0'], inplace=True)
                df_test.drop(columns=['Unnamed: 0'], inplace=True)

                # change rul to Y_THRESHOLD for everything bigger than Y_THRESHOLD. 
                # it is not important to predict these big RUL right. Only the last critical RUL are relevant
                df_train['rul'].loc[df_train['rul'] > Y_THRESHOLD] = Y_THRESHOLD
                df_test['rul'].loc[df_test['rul'] > Y_THRESHOLD] = Y_THRESHOLD
                df_test['rul'].value_counts().sort_values()

                # create min max scaler for features
                INFO = ['id', 'cycle', 'rul', 'label']
                FEATURES = df_train.drop(columns=INFO).columns
                FEATURES_ORIG = FEATURES # define the original features for FE
                scaler = MinMaxScaler()
                df_feat = pd.DataFrame(data=scaler.fit_transform(df_train[FEATURES]), columns=scaler.get_feature_names_out(), index=df_train.index)
                df_train = pd.merge(left=df_train[INFO], right=df_feat, left_index=True, right_index=True)
                df_feat_test = pd.DataFrame(data=scaler.transform(df_test[FEATURES]), columns=scaler.get_feature_names_out(), index=df_test.index)
                df_test = pd.merge(left=df_test[INFO], right=df_feat_test, left_index=True, right_index=True)

                # use the log(rul + 1)
                if LOG_Y:
                    df_train['rul'] = np.log1p(df_train['rul'])
                    df_test['rul'] = np.log1p(df_test['rul'])

                # create polynomials for features (default = False)
                run_code = False
                if run_code:
                    poly = PolynomialFeatures(include_bias=False, interaction_only=False, degree=2)
                    df_poly = pd.DataFrame(data=poly.fit_transform(df_train[FEATURES_ORIG]), columns=poly.get_feature_names_out(), index=df_train.index)
                    df_train = pd.merge(left=df_train[INFO], right=df_poly, left_index=True, right_index=True)
                    df_poly_test = pd.DataFrame(data=poly.transform(df_test[FEATURES_ORIG]), columns=poly.get_feature_names_out(), index=df_test.index)
                    df_test = pd.merge(left=df_test[INFO], right=df_poly_test, left_index=True, right_index=True)

                # create square of the features
                run_code = False
                if run_code:
                    names = [feat + '_squared' for feat in FEATURES_ORIG]
                    df_train[names] = df_train[FEATURES_ORIG] ** 2
                    df_test[names] = df_test[FEATURES_ORIG] ** 2

                run_code = False
                if run_code:
                    # add rolling variance and mean
                    df_train_mean, df_train_var = add_mean_var(df_train, DAYS_MEAN, DAYS_VAR)
                    df_test_mean, df_test_var = add_mean_var(df_test, DAYS_MEAN, DAYS_VAR)

                    # concattenate the dataframes
                    df_train = df_train.merge(df_train_mean, left_index=True, right_index=True)
                    df_train = df_train.merge(df_train_var, left_index=True, right_index=True)
                    df_test = df_test.merge(df_test_mean, left_index=True, right_index=True)
                    df_test = df_test.merge(df_test_var, left_index=True, right_index=True)

                    # drop the na which have occured due to the rolling mean/var
                    df_train.dropna(inplace=True)
                    df_test.dropna(inplace=True)

                # redifine the features as there are new features
                LABELS = ['rul']
                FEATURES = df_train.drop(columns=INFO).columns
                # create the input arrays for train and test
                X_train, y_train_int = create_input(df_train, SEQUENCE_LENGTH, FEATURES, LABELS)
                X_test, y_test_int = create_input(df_test, SEQUENCE_LENGTH, FEATURES, LABELS)
                print(X_train.shape, X_test.shape)

                # iterate over model-relevant parameters for hyper-params-optimization
                for LSTM_1_UNITS in LSTM_1_UNITS_LIST:
                    for LSTM_2_UNITS in LSTM_2_UNITS_LIST:
                        for DENSE_1_UNITS in DENSE_1_UNITS_LIST:  
                            for N_EPOCHS in N_EPOCHS_LIST:
                                for N_BATCH_SIZE in N_BATCH_SIZE_LIST:   
                                    for LSTM_DROPOUT in LSTM_DROPOUT_LIST:
                                        for DENSE_DROPOUT in DENSE_DROPOUT_LIST:
                                            for VALIDATION_SPLIT in VALIDATION_SPLIT_LIST: 
                                                print(f"LSTM_1_UNITS: {LSTM_1_UNITS} LSTM_2_UNITS: {LSTM_2_UNITS} DENSE_1_UNITS: {DENSE_1_UNITS}")
                                                #d efine the LSTM Model
                                                with device('/device:GPU:0'):
                                                    K.clear_session()
                                                    lstm_model = Sequential(
                                                        [   # 1st layer (LSTM layer)
                                                            keras.layers.LSTM(
                                                                input_shape=(SEQUENCE_LENGTH, X_train.shape[2]),
                                                                units=LSTM_1_UNITS,
                                                                return_sequences= True # gives back a y-predict for each timestep (needed for the input of LSTM-layer 2)
                                                                ),
                                                            keras.layers.Dropout(LSTM_DROPOUT),
                                                            # 2nd layer (LSTM layer)
                                                            keras.layers.LSTM(units=LSTM_2_UNITS,
                                                                return_sequences= False),
                                                            keras.layers.Dropout(LSTM_DROPOUT),
                                                            keras.layers.Dense(units=DENSE_1_UNITS, activation='relu', 
                                                                            kernel_initializer=initializers.RandomNormal(stddev=1),
                                                                            bias_initializer=initializers.Zeros()),
                                                            keras.layers.Dropout(DENSE_DROPOUT),
                                                            keras.layers.Dense(units=1, activation='relu') # just giving through the calculated value (range -inf to +inf)
                                                        ]
                                                    )    
                                                    
                                                    lstm_model.compile(loss='mse', optimizer='adam', metrics=[tfa.metrics.RSquare()])
                                                    lstm_model.summary()

                                                    # fit the LSTM model
                                                    history = lstm_model.fit(X_train, y_train_int, 
                                                                epochs=N_EPOCHS, 
                                                                batch_size=N_BATCH_SIZE, 
                                                                validation_split=VALIDATION_SPLIT, 
                                                                verbose=1, # defines if animation will be shown while training
                                                                # callbacks = [EarlyStopping(monitor='loss', min_delta=1, patience=5, verbose=0, mode='auto')]
                                                                )

                                                    # evaluate the model and print the results
                                                    score = lstm_model.evaluate(X_test, y_test_int, verbose=0)

                                                    ### Interpretation of the data
                                                    y_pred_int = lstm_model.predict(X_test)
                                                    
                                                # calculate the RMSE for all y_test below a certain theshold and check for outliers
                                                SE_0 = 0
                                                SE_1 = 0
                                                SE_2 = 0
                                                n_0 = 0
                                                n_1 = 0
                                                n_2 = 0
                                                deltas_0 = []
                                                deltas_1 = []
                                                deltas_2 = []
                                                # retransform the log y
                                                if LOG_Y:
                                                    y_test_unlog = np.expm1(y_test_int)
                                                    y_pred_int = np.expm1(y_pred_int)
                                                else:
                                                    y_test_unlog = y_test_int
                                                for y_test, y_pred in zip(y_test_unlog, y_pred_int):
                                                    # create metrics (RMSE, MEAN, STD) between 60 and 30
                                                    if y_test <= THRESHOLD_0_VAL and y_test > THRESHOLD_1_VAL:
                                                        n_0 += 1
                                                        SE_0 += np.square(y_test - y_pred)
                                                        deltas_0.append(y_test - y_pred)
                                                    if y_test <= THRESHOLD_1_VAL and y_test > THRESHOLD_2_VAL:
                                                        n_1 += 1
                                                        SE_1 += np.square(y_test - y_pred)
                                                        deltas_1.append(y_test - y_pred)
                                                    if y_test <= THRESHOLD_2_VAL:
                                                        n_2 += 1
                                                        SE_2 += np.square(y_test - y_pred)
                                                        deltas_2.append(y_test - y_pred)
                                                rmse_0 = np.sqrt(SE_0 / n_0)[0]
                                                mean_delta_0 = np.mean(deltas_0)
                                                std_delta_0 = np.std(deltas_0)
                                                outlier_max_0 = max(deltas_0)[0]
                                                outlier_min_0 = min(deltas_0)[0]
                                                rmse_1 = np.sqrt(SE_1 / n_1)[0]
                                                mean_delta_1 = np.mean(deltas_1)
                                                std_delta_1 = np.std(deltas_1)
                                                outlier_max_1 = max(deltas_1)[0]
                                                outlier_min_1 = min(deltas_1)[0]
                                                print(f"RMSE: {rmse_1}\nDELTA mean: {mean_delta_1} std: {std_delta_1}\nOUTLIERS max= {outlier_max_1} min= {outlier_min_1}")
                                                rmse_2 = np.sqrt(SE_2 / n_2)[0]
                                                mean_delta_2 = np.mean(deltas_2)
                                                std_delta_2 = np.std(deltas_2)
                                                outlier_max_2 = max(deltas_2)[0]
                                                outlier_min_2 = min(deltas_2)[0]

                                                #### write the results of this training to the log-file
                                                results = [datetime.datetime.now(),Y_THRESHOLD,SEQUENCE_LENGTH,DAYS_MEAN,DAYS_VAR,N_EPOCHS,N_BATCH_SIZE,LSTM_1_UNITS,LSTM_2_UNITS,DENSE_1_UNITS,LSTM_DROPOUT,DENSE_DROPOUT,VALIDATION_SPLIT,np.sqrt(history.history['loss'][-1]),history.history['r_square'][-1],np.sqrt(history.history['val_loss'][-1]),history.history['val_r_square'][-1],np.sqrt(score[0]),score[1],rmse_0,mean_delta_0,std_delta_0,outlier_max_0,outlier_min_0,rmse_1,mean_delta_1,std_delta_1,outlier_max_1,outlier_min_1,rmse_2,mean_delta_2,std_delta_2,outlier_max_2,outlier_min_2]
                                                csv_writer.writerow(results)
#save the log-file    
file.close()  
files.download(params_file)
#save the model if flagged
if SAVE_MODEL:
    lstm_model.save('reg_BI_model.h5')