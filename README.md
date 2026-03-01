# 📉 Telco Customer Churn Prediction

An end-to-end Machine Learning project to identify at-risk customers and understand key drivers of churn. Features a modular Python architecture, automated hyperparameter tuning, and a production-ready web interface.

## 🚀 Key Features
- **Exploratory Data Analysis (EDA):** Deep dive into churn drivers (Fiber Optic issues, Contract types, Tenure impact).
- **Inference Pipeline:** Robust data cleaning and feature engineering encapsulated in modular Python scripts.
- **Model Comparison:** Evaluated Logistic Regression vs. CatBoost (optimized via Optuna).
- **Deployment-ready:** Containerized with Docker and served via an interactive Streamlit UI.

## 📊 Results & Model Selection
| Model | Recall (Class 1) | F1-Score | Status |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **0.81** | **0.62** | **Winner** |
| **CatBoost (Tuned)** | 0.72 | 0.60 | Baseline |

**Strategic Choice:** Logistic Regression was selected for production due to its superior **Recall (0.81)** and high interpretability. In a churn scenario, identifying 81% of potential leavers is more valuable than higher overall accuracy.

## 🛠 Tech Stack
- **Language:** Python 3.12
- **Env/Package Manager:** [uv](https://github.com/astral-sh/uv)
- **ML Libraries:** Scikit-learn, CatBoost, Optuna
- **Visuals:** Seaborn, Matplotlib
- **App/Ops:** Streamlit, Docker, Joblib

## 📁 Project Structure
```text
├── data/               # Raw dataset (excluded from Git)
├── models/             # Serialized .joblib models
├── notebooks/          # EDA and Model Experiments
├── src/
│   ├── data_preprocessing.py  # Cleaning & Pipeline logic
│   ├── app.py                 # Streamlit Web UI
│   └── predict.py             # CLI Inference script
├── Dockerfile          # Container configuration
└── pyproject.toml      # Project dependencies (uv)
```

## 💻 How to Run

### Local Development (via uv)
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Run the Web App:
   ```bash
   uv run streamlit run src/app.py
   ```

### Docker
1. Build the image:
   ```bash
   docker build -t churn-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 churn-app
   ```
Access the UI at `http://localhost:8501`.

## 📈 Top Business Insights
*   **High Risk:** Customers on "Month-to-month" contracts using "Fiber optic" internet show the highest churn probability.
*   **Retention Drivers:** Long-term contracts (2-year) and "Online Security" add-ons significantly increase customer stickiness.