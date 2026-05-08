# Telecom Customer Churn Prediction

This project is an end-to-end Machine Learning product that predicts whether a telecom customer is likely to churn.

## 📌 Business Problem
High customer churn negatively impacts revenue and growth. This project aims to identify customers at high risk of churn, enabling personalized retention offers and improving customer lifetime value (CLV).

## 📂 Project Structure
- `data/`: Contains the customer churn dataset.
- `notebooks/`: Jupyter Notebooks for EDA, Modeling, and Explainability.
- `src/`: Source code for model training.
- `models/`: Pickled model and preprocessing artifacts.
- `api/`: Flask REST API to serve predictions.
- `dashboard.py`: Streamlit dashboard for interactive predictions.
- `requirements.txt`: Python dependencies.

## 📊 Dataset Description
The dataset includes demographics, account details, services subscribed, and billing information.
Target variable: `Churn` (Yes / No).

## 🧠 Modeling Approach
1. **EDA**: Explored distributions and relationships. Visualized class imbalance and important feature relationships like tenure and churn.
2. **Feature Engineering**: Created features like `AvgMonthlySpend`, `TenureBucket`, and `ServiceCount`.
3. **Class Imbalance**: Used SMOTE to balance the dataset.
4. **Model Building**: Trained a Random Forest classifier.
5. **Threshold Optimization**: Optimized probability threshold to maximize F1-Score instead of using the default 0.5.
6. **Explainability**: Used SHAP for global feature importance and local predictions.

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train.py
```
This will train the model and save the required files to the `models/` directory.

### 3. Run the Flask API
```bash
cd api
python app.py
```
The API will run on `http://127.0.0.1:5000/predict`

### 4. Run the Streamlit Dashboard (Bonus)
In a new terminal window:
```bash
streamlit run dashboard.py
```

## 📈 Business Insights
- New customers are the most likely to churn. Focus on better onboarding experiences.
- Customers on month-to-month contracts have the highest churn rate. Incentivize moving to one-year or two-year contracts.
- Fiber optic internet users churn more frequently than DSL users; technical issues or pricing of fiber optic could be investigated.
