### Capital Bikeshare - Kaggle Competition

#### target: 
to predict the hourly demand of shared bikes using time and weather features.

#### used models:
Linear Regression models with and without Ridge loss function.

#### files:
- 3_1_datetime: in this file I create new time features out of datetime and perform an EDA
- 3_3_lin_reg_feat: in this file I perform some additional EDA, feature selection 
and feature engineering (like MinMax-Scaling, Repeating Basis Functions, Sinus-/Cosine-Modelling, 
polynomial features creation and one hot encoding).
Finally I evaluate the data with 2 linear regression models (with and without Ridge-Function)
- 3_5_lin_reg_pipe: taking the best data from the first 2 files, I create a pipeline with 
data preprocessing (via transformers, pipelines and a columntransformer) and a ridge model
I use that pipeline to perform GridSearch hyperparameter optimization.
- 3_7_final_model: with the best parameters resulting from the former hyperparameter optimization I 
create a final pipeline, predict the target variable for the test data and create an upload 
file to Kaggle.

#### result: 
position 500 out of 3200 (competition was already closed)