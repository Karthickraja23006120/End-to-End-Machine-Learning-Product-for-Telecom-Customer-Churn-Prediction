from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Telecom Churn Prediction API")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    InternetService: str
    Contract: str
    MonthlyCharges: float

    # Note: providing defaults for other missing features from the sample input
    # Since the input example lacks some features, we will set them to "No"
    MultipleLines: str = "No"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"

# Load models and preprocessors globally
model = None
scaler = None
label_encoders = None
columns = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_encoders, columns
    try:
        with open('../models/xgboost_best.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('../models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('../models/columns.pkl', 'rb') as f:
            columns = pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {str(e)}")

@app.get("/")
def home():
    return {"message": "Welcome to Telecom Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model implies not loaded")
    
    # Calculate derived features
    TotalCharges = data.MonthlyCharges * data.tenure
    AvgMonthlySpend = TotalCharges / data.tenure if data.tenure > 0 else 0
    
    # Tenure bucket
    if data.tenure <= 12:
        Tenure_Bucket = "New"
    elif data.tenure <= 48:
        Tenure_Bucket = "Medium"
    else:
        Tenure_Bucket = "Loyal"

    # Contract LongTerm
    Is_LongTerm_Contract = 1 if data.Contract in ['One year', 'Two year'] else 0
    
    # Setup dataframe
    input_dict = data.dict()
    input_dict['TotalCharges'] = TotalCharges
    input_dict['AvgMonthlySpend'] = AvgMonthlySpend
    input_dict['Tenure_Bucket'] = Tenure_Bucket
    input_dict['Is_LongTerm_Contract'] = Is_LongTerm_Contract
    
    # Services
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_dict['Service_Count'] = sum([1 if input_dict[svc] == 'Yes' else 0 for svc in services])
    input_dict['High_Value_Customer'] = 1 if data.MonthlyCharges >= 89.85 else 0 # 89.85 based on sample

    df = pd.DataFrame([input_dict])

    # Encode categoricals using loaded label encoders array
    for col, le in label_encoders.items():
        if col in df.columns:
            # handle unseen labels
            val = str(df.iloc[0][col])
            val = val if val in le.classes_ else le.classes_[0]
            df[col] = le.transform([val])

    # Scale numeric
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'Service_Count']
    df[num_cols] = scaler.transform(df[num_cols])

    # Ensure columns order matches test data
    df = df[columns]

    prob = float(model.predict_proba(df)[0, 1])
    # The requirement optimal threshold can be set. Assume 0.5.
    churn = "High" if prob >= 0.5 else "Low"

    return {
        "churn_probability": round(prob, 2),
        "churn_risk": churn
    }
