import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models using st.cache_resource
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load models with better names for clarity
logistic_model = load_model('Logistic_Regression.pkl')
xgb_model = load_model('XGBoost_Classifier.pkl')
svc_model = load_model('Support_Vector_Classifier.pkl')

# Streamlit UI
st.title('Credit Card Fraud Detection')

st.sidebar.header('Input Features')
st.sidebar.write("Enter the values for the features below to make a prediction.")

# Define feature columns (adjust if necessary)
cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Create a dictionary for user inputs
user_input = {}
for col in cols:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

# Convert dictionary to DataFrame
input_data = pd.DataFrame([user_input])

# Ensure that input_data has the same feature columns as used in training
input_data = input_data.reindex(columns=cols, fill_value=0)

if st.sidebar.button('Predict'):

    try:
        # Make prediction with the best model
        xgb_pred = xgb_model.predict(input_data)[0]

        # Display results
        st.subheader('Prediction Results:')
        st.write(f' {"Fraud" if xgb_pred == 1 else "Not Fraud"}')

    except ValueError as e:
        st.error(f"Error: {e}")
        st.write("Please check the feature inputs and ensure they match the expected format.")
