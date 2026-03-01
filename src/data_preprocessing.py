import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df = df.drop_duplicates()

    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: "No", 1: "Yes"})

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset='TotalCharges')

    cols_with_internet = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in cols_with_internet:
        df[col] = df[col].replace('No internet service', 'No')

    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
    return df

def get_preprocessing_pipeline(num_features, cat_features):
    num_transformer = StandardScaler()

    cat_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )

    preprocessor = ColumnTransformer(transformers=[
        ('num', cat_transformer, cat_features),
        ('cat', num_transformer, num_features)
    ])

    return preprocessor