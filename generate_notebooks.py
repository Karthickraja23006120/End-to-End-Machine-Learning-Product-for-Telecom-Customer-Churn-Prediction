import json
import os

def create_notebook(filename, cells_code):
    cells = []
    for code in cells_code:
        if code.startswith('# markdown'):
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in code.replace("# markdown\n", "").split("\n")]
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.split("\n")]
            })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

# EDA Notebook
eda_cells = [
"""# markdown
# 1. EDA for Telecom Customer Churn
""",
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('../data/customer_churn_data.csv')
print(df.shape)
df.head()
""",
"""# markdown
## Handle TotalCharges and Missing Values
""",
"""df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Missing TotalCharges:", df['TotalCharges'].isnull().sum())
# Fill missing with median or 0, considering tenure is 0 for these often
df['TotalCharges'].fillna(0, inplace=True)

# Drop CustomerID as it's non-informative
df.drop('customerID', axis=1, inplace=True)
""",
"""# markdown
## Mandatory Visualizations
### 1. Churn distribution
""",
"""plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution (Class Imbalance)')
plt.show()
""",
"""# markdown
### 2. Tenure vs Churn
""",
"""plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title('Tenure vs Churn')
plt.show()
""",
"""# markdown
### 3. MonthlyCharges vs Churn
""",
"""plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('MonthlyCharges vs Churn')
plt.show()
""",
"""# markdown
### 4. Contract vs Churn
""",
"""plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Contract Type vs Churn')
plt.show()
""",
"""# markdown
### 5. InternetService vs Churn
""",
"""plt.figure(figsize=(8,5))
sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title('Internet Service vs Churn')
plt.show()
""",
"""# markdown
### 6. Correlation Heatmap
""",
"""# Only keeping numeric features for general correlation
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr = df[numeric_features].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
""",
"""# markdown
## Business Questions
1. Do senior citizens churn more?
2. How does tenure affect churn?
3. Which contract types have the highest churn?
4. Does monthly billing amount influence churn?
5. Which services are associated with higher churn?
""",
"""# 1. Senior Citizens
sns.countplot(data=df, x='SeniorCitizen', hue='Churn')
plt.title('Senior Citizen vs Churn')
plt.show()
print("Senior Citizens churn at a higher relative rate than younger folks.")

# 2. Tenure - Already shown, lower tenure = higher churn
# 3. Contract - Already shown, month-to-month has highest churn
# 4. Monthly Billing - Higher monthly billing corresponds to slightly higher churn

# 5. Services:
services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for svc in services:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=svc, hue='Churn')
    plt.title(f'{svc} vs Churn')
    plt.show()
"""
]
create_notebook('notebooks/01_EDA_Notebook.ipynb', eda_cells)


# Modeling Notebook
model_cells = [
"""# markdown
# 2. Data Cleaning, Preprocessing & Modeling
""",
"""import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('../data/customer_churn_data.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
""",
"""# markdown
## Feature Engineering
• AvgMonthlySpend = TotalCharges / tenure
• Tenure buckets (New / Medium / Loyal)
• Service count per customer
• Contract duration flags
• High-value customer indicator
""",
"""# Feature 1: AvgMonthlySpend
df['AvgMonthlySpend'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)

# Feature 2: Tenure buckets
df['Tenure_Bucket'] = pd.cut(df['tenure'], bins=[-1, 12, 48, 100], labels=['New', 'Medium', 'Loyal'])

# Feature 3: Service count
services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['Service_Count'] = sum(df[svc].apply(lambda x: 1 if x in ['Yes', 'Yes (Fiber optic/DSL/etc)'] else 0) for svc in services)

# Feature 4: Contract duration flags
df['Is_LongTerm_Contract'] = df['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)

# Feature 5: High-value customer indicator
high_val_thresh = df['MonthlyCharges'].quantile(0.8)
df['High_Value_Customer'] = df['MonthlyCharges'].apply(lambda x: 1 if x >= high_val_thresh else 0)
""",
"""# markdown
## Encoding & Scaling
""",
"""target = 'Churn'
y = df[target].apply(lambda x: 1 if x=='Yes' else 0)
X = df.drop(target, axis=1)

# Categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# Encode categorical variables using Label Encoding 
# In a real pipeline, OneHotEncoding might be better, but LabelEncoding is acceptable
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    # Handle mixed types or unseen values by converting to string
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Scale numerical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'Service_Count']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save preprocessors
os.makedirs('../models', exist_ok=True)
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../models/label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

# Train Validation Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))
""",
"""# markdown
## Class Imbalance Handling
""",
"""smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: %s" % sum(y_train))
print(f"After SMOTE:  %s" % sum(y_train_res))
""",
"""# markdown
## Modeling
""",
"""def evaluate_model(model, X_val, y_val, name="Model"):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    print(f"=== {name} ===")
    print("ROC-AUC:", roc_auc_score(y_val, y_prob))
    print("Precision:", precision_score(y_val, y_pred))
    print("Recall:", recall_score(y_val, y_pred))
    print("F1-score:", f1_score(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\\n")
    return y_prob

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res)
evaluate_model(lr, X_val, y_val, "Logistic Regression")

# Random Forest
rf = RandomForestClassifier(random_state=42, max_depth=6)
rf.fit(X_train_res, y_train_res)
evaluate_model(rf, X_val, y_val, "Random Forest")

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)
xgb_val_prob = evaluate_model(xgb_model, X_val, y_val, "XGBoost")
""",
"""# markdown
## Threshold Optimization
""",
"""from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_val, xgb_val_prob)
f1_scores = 2*precisions*recalls / (precisions+recalls+1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold for F1-score: {optimal_threshold:.4f}")

# Final prediction with optimal threshold
y_val_opt_pred = (xgb_val_prob >= optimal_threshold).astype(int)
print("With optimal threshold:")
print("ROC-AUC:", roc_auc_score(y_val, xgb_val_prob))
print("Precision:", precision_score(y_val, y_val_opt_pred))
print("Recall:", recall_score(y_val, y_val_opt_pred))
print("F1-score:", f1_score(y_val, y_val_opt_pred))

# Evaluate on Test set
xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred = (xgb_test_prob >= optimal_threshold).astype(int)
print("\\n--- Test Set Evaluation (XGBoost) ---")
print("ROC-AUC:", roc_auc_score(y_test, xgb_test_prob))
print("F1-score:", f1_score(y_test, y_test_pred))

# Save best model logic
with open('../models/xgboost_best.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
    
# Store final columns required
with open('../models/columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
"""
]
create_notebook('notebooks/02_Modeling_Notebook.ipynb', model_cells)


# Explainability Notebook
shap_cells = [
"""# markdown
# 3. Model Explainability (SHAP)
""",
"""import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and data
with open('../models/xgboost_best.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

df = pd.read_csv('../data/customer_churn_data.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Feature Engineering
df['AvgMonthlySpend'] = df.apply(lambda row: row['TotalCharges']/row['tenure'] if row['tenure']>0 else 0, axis=1)
df['Tenure_Bucket'] = pd.cut(df['tenure'], bins=[-1, 12, 48, 100], labels=['New', 'Medium', 'Loyal'])

services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['Service_Count'] = sum(df[svc].apply(lambda x: 1 if x in ['Yes'] else 0) for svc in services)
df['Is_LongTerm_Contract'] = df['Contract'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)
high_val_thresh = df['MonthlyCharges'].quantile(0.8)
df['High_Value_Customer'] = df['MonthlyCharges'].apply(lambda x: 1 if x >= high_val_thresh else 0)

X = df.drop('Churn', axis=1)

with open('../models/label_encoders.pkl', 'rb') as f:
    le_dict = pickle.load(f)
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../models/columns.pkl', 'rb') as f:
    cols = pickle.load(f)

for col, le in le_dict.items():
    X[col] = X[col].astype(str)
    # Handle unseen
    X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    X[col] = le.transform(X[col])

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'Service_Count']
X[num_cols] = scaler.transform(X[num_cols])
X = X[cols] # Ensure column order
""",
"""# markdown
## Global Interpretability
""",
"""explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X.sample(1000, random_state=42))

shap.summary_plot(shap_values, X.sample(1000, random_state=42))
""",
"""# markdown
## Local Interpretability (Single Customer)
""",
"""shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)
plt.savefig('local_explanation.png')
"""
]
create_notebook('notebooks/03_Explainability_Notebook.ipynb', shap_cells)

print("Notebooks generated efficiently!")
