import pandas as pd
import numpy as np

def load_data(filepath='../data/customer_churn_data.csv'):
    df = pd.read_csv(filepath)
    df.drop('customerID', axis=1, inplace=True, errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

def feature_engineering(df):
    df['AvgMonthlySpend'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)
    df['Tenure_Bucket'] = pd.cut(df['tenure'], bins=[-1, 12, 48, 100], labels=['New', 'Medium', 'Loyal'])
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = sum(df[svc].apply(lambda x: 1 if x in ['Yes', 'Yes (Fiber optic/DSL/etc)'] else 0) for svc in services)
    df['Is_LongTerm_Contract'] = df['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)
    high_val_thresh = df['MonthlyCharges'].quantile(0.8)
    df['High_Value_Customer'] = df['MonthlyCharges'].apply(lambda x: 1 if x >= high_val_thresh else 0)
    return df
