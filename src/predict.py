import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_data


def predict_customer_churn(customer_data_dict, model_path='models/final_churn_model.joblib'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}.")

    model = joblib.load(model_path)

    df_raw = pd.DataFrame([customer_data_dict])

    df_cleaned = clean_data(df_raw)

    prediction = model.predict(df_cleaned)[0]
    probability = model.predict_proba(df_cleaned)[0][1]

    return {
        'churn_prediction': 'Yes' if prediction == 1 else 'No',
        'churn_probability': round(float(probability), 4)
    }


if __name__ == "__main__":
    example_customer = {
        'customerID': '7590-VHVEG',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': '29.85'
    }

    print("--- Прогноз для нового клиента ---")
    result = predict_customer_churn(example_customer)
    print(f"Результат: {result['churn_prediction']}")
    print(f"Вероятность оттока: {result['churn_probability'] * 100}%")