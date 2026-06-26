# 👥 AttritionIQ — HR Attrition Prediction & Analytics

> End-to-end HR analytics project predicting employee attrition using Logistic Regression & XGBoost with SHAP explainability, risk scoring, and an interactive Streamlit dashboard.

---
## Live Demo
https://hr-attrition-prediction-analytics-hugyj5bxdwtuym5tvnkpdf.streamlit.app/

## 🎯 Business Problem
Companies lose $50,000–$200,000 per employee who leaves due to hiring and training costs.
This project answers:
- **Which employees are most likely to leave?**
- **What are the top drivers of attrition?**
- **Which departments/roles need urgent HR intervention?**

---

## 🔑 Key Results
- XGBoost achieved **89% AUC** on test set
- Identified **OverTime, MonthlyIncome, JobSatisfaction** as top 3 attrition drivers via SHAP
- Generated individual **attrition risk scores** for all 1,470 employees
- Interactive dashboard enabling HR teams to filter by department, role, salary band

---

## 🏗️ Architecture
```
IBM HR Dataset (Kaggle)
        ↓
EDA & Visualization (Pandas, Seaborn, Plotly)
        ↓
Feature Engineering + Encoding
        ↓
Model Training (Logistic Regression + XGBoost)
        ↓
SHAP Explainability (Global + Local)
        ↓
Attrition Risk Scoring
        ↓
Streamlit Dashboard
```

---

## 🛠️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |

---

## 📁 Project Structure
```
hr_attrition/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   ├── 01_EDA.py
│   ├── 02_preprocessing.py
│   ├── 03_logistic_regression.py
│   ├── 04_xgboost_model.py
│   └── 05_risk_scoring.py
├── sql/
│   └── queries.sql
├── dashboard/
│   └── app.py
└── requirements.txt
```

---

## ⚙️ How to Run

### 1. Download Dataset
👉 https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Place CSV in `data/` folder.

### 2. Install & Run
```bash
pip install -r requirements.txt
python notebooks/01_EDA.py
python notebooks/02_preprocessing.py
python notebooks/03_logistic_regression.py
python notebooks/04_xgboost_model.py
python notebooks/05_risk_scoring.py
streamlit run dashboard/app.py
```
