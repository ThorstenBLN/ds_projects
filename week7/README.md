### Time Series Analysis

#### data: 
daily temperature data from Tempelhofer Feld, Berlin, Germany in pandas (downloaded from www.ecad.eu)

### files:
- 7.1_project.ipynb
- 7.2_project_full.ipynb
- 7.3_ARIMA_and_DECOMPOSE

- 7.1_project.ipynb & 7.2_project_full.ipynb:
as 7.1 and 7.2 are partially overlapping, I will explain them together:
the target of both files was predict the temperature of the timeseries by creating a model 
(seperately file 7.1 or combined file 7.2) for predicting the trend, the saisonality and the 
stationary remainder. First time features were created and the missing data imputed (in the year 1945 
was no data recorded, probably due to the impact of WW2). 
The trend was predicted by a linear model with and without polynomial features. 
From the remaining temperature the saisonal effect, predicted by a linear model based on one hot
encoded month features, was deducted to receive the remainder. After proving the stationarity of the
remainder timelagged features of the remainder were created. With these timelagged features the 
remainder was predicted. The models have been evaluated by cross-validation and the prediction of 1 
day in future.

- 7.3._ARIMA_and_DECOMPOSE:
in this file an AutoArima model should predict the timeseries. Unfortunately
the notebook runs out of memory when starting with the 2nd model.
A manual ARIMA model was fitted to show the approach.
