### predictive maintenance - a jet engine failure prediction system

#### target: 
Predict the failure of an engine in advance by analyzing given sensor data. 
Hence increase of security and reduction maintenance/repair costs for airlines.

#### data: 
- challenge data for the Prognostics and Health Management data competition
- C-MAPSS simulation of 704 plane engines (90.000 lb-thrust class)
- 704 Timeseries of 21 sensors which have been placed inside the engine
- simulation models engine performance degradation due to wear and tear
- the data consisted of 4 files. As file 1 & 3 had similary engine settings (setting 1 & 2) and files 2 & 4 had also similar engine settings (setting 3). Hence I created 2 datasets and trained 2 different models for each dataset.

#### approach: 
1. engine maintance classifier with 3 classes:
- class 0: maintenance not necessary, class 1: maintenance necessary, class 2: maintenace urgent (the classes reflect the remaining useful life (RUL) of the engine)
- given the sensordata for the engines an LSTM-model is searching for patterns in that data in order to predict the RUL-class
2. engine maintenance early warning system regression model:
- given the sensordata for the engines an LSTM-model is searching for patterns in that data in order to predict the remaining useful life of the engine, once the RUL is getting lower than 60 cycles
3. a streamlit simulation in which the user can choose a test engine. For the chosen engine the sensor data is plotted, the real RUL and the predictions of the models are shown.

#### techstack: 
pandas, numpy, matplotlib, seaborn, sklearn, tensorflow keras, streamlit, plotly, gtts, playsound, google colab

#### Files (as the project consisted of many scripts here are some example scripts): 
- 1_EDA.ipnyb: 
    - adding of the labels (RUL and RUL-classes)
    - plotting and analyzation of the sensor data
    - choice of sensors as input data for the LSTM-model
    - save the edited data into csv files
- 04_reg_col_hyp.py: 
    - script for the hyperparameter optimization of the data and the LSTM on google colab
    - the script contains also some features engineering as e.g. adding rolling means and       variances of the X-features
    - all kind of data and model parameters were tested in order to find the best fitting LSTM-regression model
- 05_cat_col_FINAL.py:
    - script for the training of final model after the hyperparameter optimization for the classification model (analogue to 04_reg_col_hyp.py)
    - also here rolling means and variances have been added, a LSTM-model has been fitted and 
    the fitted model has been saved
- engine_app_final.py:
    - this script contains a streamlit presentation of my project
    - the user can select an engine from the test data and start the engine simulation
    - on the simulation screen you can see the sensors of the engine, the real RUL and my 2 models prediction the maintenance class and the RUL (once it gets lower than 60 cycles)
    - the models will raise an alarm (visially and acustically) if the models predict the enclosing of the engine failure
    - in order to have a smoother simulation process, I created and saved the prediction data for all engines in advance. In the simulation I used these x-test and y-predict numpy arrays.
- text.py:
    - this script creates the audio files for the acustical warnings with gtts.
    
#### results:
- dataset 1:
    - classification: accuracy, precision, recall ca. 99%, 
    - regression: R2 score 97%, RMSE total: 3,9 cycles
- dataset 2:
    - the results have been slightly worse than for dataset 1 as patterns of deteroriation were harder to find for setting 3 of the engine. Nevertheless the accuracy, precision and recall of ca. 96% was reached.