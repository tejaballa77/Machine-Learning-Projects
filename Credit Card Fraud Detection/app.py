import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define feature names
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Input fields
st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details:")

input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Create DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# ✅ Ensure same scaling as training
scaled_input = scaler.transform(input_df)

# Predict
prediction = model.predict(scaled_input)[0]

# Display result
if prediction == 1:
    st.error("🚨 Fraudulent Transaction Detected!")
else:
    st.success("✅ Legitimate Transaction")
