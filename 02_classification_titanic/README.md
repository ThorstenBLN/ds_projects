## Titanic Survivor Prediction  
#### content: 
This project is about the Kaggle Titanic survivor competition  
#### Target: 
to predict passenger survival with machine learning models using passenger features like Age, Sex, Class or Fare.  
#### files:  
1.) 2_6_Feature_engineering.ipnyb:
- I explore the data and perform feature engineering
- Then I choose 4 classifaction models (Log. Regression, Forest Tree, Random Forest and Support Vector Machines)
  to predict the survivors
- I compare the models with confusion matrices, ROC-curves and precision-/recall-curves

2.) 2_7_FE_with_Columntransformers.ipnyb:
- I took the best model from file 1 (random forest classifier) and created a datapipeline for it.
- the pipeline consists of a Columntransformer with all data preprocessors (transformers or 
  transformer pipelines)
- I trained the model, made predictions for the test data and prepare the upload file for the Kaggle
  competition 
