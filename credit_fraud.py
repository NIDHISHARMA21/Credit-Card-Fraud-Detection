
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import pickle

# DATASET LINK: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

df = pd.read_csv(r"E:\2_internship with Codsoft\5_credit_card_fraud_project\creditcard.csv")

#Checking Missing Values
df.isnull().sum()

value_counts = df['Class'].value_counts()
print(value_counts)

winsor_df = pd.DataFrame()
cols = df.drop(['Time','Class'], axis = 1).columns

for col in cols:
    # Boxplot Before Winsorization
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel(col)
    plt.boxplot(df[col])
    plt.title('Before Winsorization')
    
    # Boxplot After Winsorization
    winsor = Winsorizer(capping_method = 'quantiles',
                        tail = 'both',
                        fold = 0.05)
    winsor_df[col] = winsor.fit_transform(df[[col]])
    
    plt.subplot(1, 2, 2)
    plt.xlabel(col)
    plt.boxplot(winsor_df[col])
    plt.title('After Winsorization')
    plt.show()
    
for col in ['V6','V20','V27','V28','Amount']:
    # Boxplot Before Winsorization
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel(col)
    plt.boxplot(winsor_df[col])
    plt.title('Before Winsorization')
    
    # Boxplot After Winsorization
    winsor = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 0.05)
    winsor_df[col] = winsor.fit_transform(df[[col]])
    
    plt.subplot(1, 2, 2)
    plt.xlabel(col)
    plt.boxplot(winsor_df[col])
    plt.title('After Winsorization')
    plt.show()

# Box Plot
for col in cols:
    sns.boxplot(winsor_df[col])
    plt.show()    

X = pd.concat([df['Time'], winsor_df], axis=1)
y = df['Class']

# Heatmap for correlation matrix
plt.figure(figsize= (20,15))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.show()
    
# Histograms for visualizing distributions
plt.figure(figsize= (1920,1080))
X.hist(bins=30, figsize=(10, 10))
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Define Models
models = {
    'Logistic Regression': LogisticRegression(),
    'XGBoost Classifier': XGBClassifier(),
    'Support Vector Classifier': SVC(kernel='rbf')
}

# Iterate over models and evaluate them
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    train_auc = metrics.roc_auc_score(y_train, y_pred_train)
    test_auc = metrics.roc_auc_score(y_test, y_pred_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)
    
    # Print the evaluation metrics
    print(f'{name} - Training AUC: {train_auc:.4f}')
    print(f'{name} - Validation AUC: {test_auc:.4f}')
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)
    print("############################################################\n")
    
    # Save the model using pickle
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{name.replace(" ", "_")}_{timestamp}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model {name} saved as {filename}')

print("Model training and evaluation completed.")



