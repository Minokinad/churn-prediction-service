import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import clean_data

st.set_page_config(page_title="Churn Analysis Tool", page_icon="📈", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load('models/final_churn_model.joblib')


model = load_model()

st.title("📊 Customer Churn Prediction Dashboard")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, value=1)

    with col2:
        st.header("🌐 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.header("💳 Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)

if st.button("🚀 Run Prediction Analysis", use_container_width=True):
    input_dict = {
        'customerID': 'inference_id',
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': protection,
        'TechSupport': support,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': str(total)
    }

    df_input = pd.DataFrame([input_dict])

    df_processed = clean_data(df_input)

    probability = model.predict_proba(df_processed)[0][1]

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("Prediction Result")
        if probability > 0.5:
            st.error(f"**High Risk of Churn**")
        else:
            st.success(f"**Customer is Loyal**")

    with res_col2:
        st.subheader("Churn Probability")
        st.progress(probability)
        st.write(f"The probability of churn is **{probability * 100:.1f}%**")