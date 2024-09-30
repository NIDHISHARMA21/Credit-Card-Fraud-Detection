# Credit-Card-Fraud-Detection

##### This project involves detecting fraudulent credit card transactions using various machine learning models. The dataset used is the Credit Card Fraud Detection dataset, which contains anonymized features of transactions and a target variable indicating fraud.

#### Key Steps:

#### Data Preprocessing:

###### Loaded the dataset and checked for missing values.
###### Applied Winsorization to handle outliers in the features.
###### Performed exploratory data analysis (EDA) with visualizations such as box plots, heatmaps, and histograms to understand the data distribution and correlations.

#### Model Training:
###### Split the data into training and testing sets.
###### Trained and evaluated three models: Logistic Regression, XGBoost Classifier, and Support Vector Classifier (SVC).
###### Evaluated models using metrics like AUC, accuracy, confusion matrix, and classification report.

#### Model Saving:

###### Saved the trained models using Pickle for future use.

#### Tools & Libraries:
Python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
pickle
###### This repository includes the code for data preprocessing, model training, evaluation, and saving the models. The goal is to provide a robust solution for detecting fraudulent transactions effectively.
