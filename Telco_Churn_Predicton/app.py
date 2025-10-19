# app.py
import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
model_path = '/Users/b.tejateja/Downloads/Telco_Churn_App/logistic_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# App title
st.title("📊 Telco Customer Churn Prediction App")
st.write("Predict if a customer will churn or not.")

# Sidebar inputs
st.sidebar.header("Enter customer details:")

gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
gender_num = 1 if gender == 'Male' else 0

SeniorCitizen = st.sidebar.selectbox('Senior Citizen', (0,1))
tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 12)
MonthlyCharges = st.sidebar.number_input('Monthly Charges', 0, 1000, 70)
TotalCharges = st.sidebar.number_input('Total Charges', 0, 10000, 500)
HasInternetService = st.sidebar.selectbox('Has Internet Service?', (True, False))
HasInternetService_num = 1 if HasInternetService else 0

# Create input dataframe
input_data = pd.DataFrame({
    'gender': [gender_num],
    'SeniorCitizen': [SeniorCitizen],
    'tenure': [tenure],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'HasInternetService': [HasInternetService_num]
})

st.subheader("Customer Input")
st.dataframe(input_data)

# Make prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader("Prediction")
st.write("Customer will churn" if prediction[0] == 1 else "Customer will NOT churn")

st.subheader("Prediction Probability")
st.write(prediction_proba)
