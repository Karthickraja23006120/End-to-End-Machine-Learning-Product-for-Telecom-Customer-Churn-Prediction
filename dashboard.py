import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

st.title("Telecom Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict if they are likely to churn.")

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)

st.sidebar.header("Services Subscribed")
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_prot = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.sidebar.header("Billing Information")
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=50.0)

if st.button("Predict Churn"):
    # Construct input payload
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    try:
        # Request prediction from API
        res = requests.post("http://127.0.0.1:5000/predict", json=payload)
        
        if res.status_code == 200:
            result = res.json()
            prob = result.get("churn_probability")
            risk = result.get("churn_risk")
            
            st.subheader("Prediction Result")
            if risk == "High":
                st.error(f"High Risk of Churn! Probability: {prob}")
            else:
                st.success(f"Low Risk of Churn. Probability: {prob}")
        else:
            st.error("Error connecting to the API. Make sure the Flask API is running on port 5000.")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
