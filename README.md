# End-to-End Machine Learning Product for Telecom Customer Churn Prediction

## Business Problem
The telecom company is facing a high customer churn rate, which negatively impacts revenue and growth. Retaining existing customers is often more cost-effective than acquiring new ones. The goal of this project is to build an end-to-end Machine Learning product that predicts whether a customer will churn, explains the reasons driving the churn, and enables the business to take proactive retention actions through a production-ready API.

## Dataset Description
The historical customer dataset contains the following attributes:
- **Demographics:** Gender, SeniorCitizen status, Partner, Dependents.
- **Account Details & Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Tenure.
- **Billing Information:** Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
- **Target Variable:** Churn (Yes / No) - Indicates whether the customer left within the last month.

## Modeling Approach
1. **Data Understanding & Preprocessing:**
   - Addressed missing values in `TotalCharges`.
   - Dropped non-informative columns like `customerID`.
   - Transformed categorical columns with Label Encoding.
   - Scaled numerical features using standard scaling.

2. **Feature Engineering:**
   - Created `AvgMonthlySpend` based on `TotalCharges` / `tenure`.
   - Engineered `Tenure_Bucket` dividing tenure into New, Medium, and Loyal categories.
   - Counted active services to create `Service_Count`.
   - Identified long-term contract users (`Is_LongTerm_Contract`) and high-spending customers (`High_Value_Customer`).

3. **Handling Class Imbalance & Modeling:**
   - Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   - Trained Baseline Logistic Regression, Random Forest, and XGBoost models.
   - Selected XGBoost as the best-performing model based on F1-Score, ROC-AUC, Precision, and Recall.
   - Evaluated optimal thresholds to favor better recall or F1 metrics to maximize business value given the cost of customer acquisition.

4. **Model Explainability:**
   - Utilized SHAP (SHapley Additive exPlanations) to interpret the model globally and locally, providing transparency on which features influence a customer's likelihood of churning.

## API Usage
The model is deployed as a REST API using **FastAPI**.

### Running the API
From the project root directory, launch the API Server:
```bash
cd api
uvicorn app:app --reload
```
The API is exposed locally at `http://127.0.0.1:8000`. 
Interactive Swagger Documentation can be found at `http://127.0.0.1:8000/docs`.

### Making a Prediction
**Endpoint:** `/predict`  
**Method:** `POST`

**Sample JSON Input:**
```json
{
 "gender": "Female",
 "SeniorCitizen": 0,
 "Partner": "Yes",
 "Dependents": "No",
 "tenure": 5,
 "PhoneService": "Yes",
 "InternetService": "Fiber optic",
 "Contract": "Month-to-month",
 "MonthlyCharges": 89.85
}
```

**Sample Output:**
```json
{
 "churn_probability": 0.81,
 "churn_risk": "High"
}
```

## Business Insights
Through Exploratory Data Analysis and SHAP values, several crucial insights were derived:
- **Tenure:** Newer customers have a significantly higher churn rate compared to loyal customers.
- **Contracts:** Customers on "Month-to-month" contracts are far more likely to churn than those on "One year" or "Two year" commitments.
- **Internet Service:** Fiber optic users display higher churn despite the premium service, which suggests either competition or pricing dissatisfaction.
- **Support Services:** The absence of services like TechSupport or OnlineSecurity contributes heavily to customer departure. Encouraging add-on services can strengthen retention.
- **Billing Amount:** Higher monthly charges tend to moderately correlate with higher churn, demanding competitive pricing or perceived value adjustments for premium tier customers.
