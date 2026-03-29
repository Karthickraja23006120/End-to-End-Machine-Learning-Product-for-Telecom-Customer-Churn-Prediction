import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import sys

# Assume run from project root
sys.path.append(os.path.abspath('.'))
from src.data_processing import load_data, feature_engineering
from utils.metrics import evaluate

def train_model():
    df = load_data('data/customer_churn_data.csv')
    df = feature_engineering(df)
    
    target = 'Churn'
    y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop(target, axis=1)
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'Service_Count']
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
    with open('models/columns.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_res, y_train_res)
    
    evaluate(xgb_model, X_test, y_test, "XGBoost Test Performance")
    
    with open('models/xgboost_best.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
        
if __name__ == "__main__":
    train_model()
