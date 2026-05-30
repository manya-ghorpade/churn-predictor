# Customer Churn Predictor

An end-to-end Machine Learning project that predicts whether a telecom customer will churn, with a SHAP explainability layer and an interactive Streamlit UI.

---

## Project Structure

```
churn-predictor/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw dataset
│   ├── telco_churn_clean.csv                   # cleaned dataset
│   ├── churn_distribution.png
│   └── correlation_heatmap.png
├── notebooks/
│   |── EDA.ipynb
|   |── models.ipynb
|   |── SHAP.ipynb                             # Week 1 — EDA
├── app.py                                      # Streamlit app (Week 4)
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn
- **Explainability:** SHAP
- **UI + Deployment:** Streamlit, Streamlit Cloud

---

## Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7032 customers, 20 features after cleaning
- Binary target: `Churn` (1 = churned, 0 = retained)
- Class imbalance: 73% No Churn, 27% Churn

---

## Week 1 — Exploratory Data Analysis

**Key findings:**
- 26.5% churn rate — class imbalance addressed in modeling
- Customers on month-to-month contracts churn at 43% vs 3% on two-year contracts
- Median tenure of churned customers is ~10 months vs ~38 months for retained
- Higher monthly charges correlate with churn; fiber optic customers churn at 42%
- `TotalCharges` and `tenure` are highly collinear (r=0.83)
- `gender` has near-zero predictive value (-0.009)

**Top features correlated with Churn:**

| Feature | Correlation |
|---|---|
| Contract | -0.396 |
| tenure | -0.354 |
| OnlineSecurity | -0.289 |
| TechSupport | -0.282 |
| MonthlyCharges | +0.193 |

---

## Week 2 — ML Models 

**Models trained:** Logistic Regression, Random Forest (scikit-learn Pipeline)

**Final model:** Logistic Regression with class_weight="balanced"

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Churn Recall | 0.79 | 0.49 |
| Churn Precision | 0.49 | 0.63 |
| Churn F1 | 0.61 | 0.56 |
| ROC-AUC | **0.8353** | 0.8179 |

**Key insight:** LR outperformed RF on recall and AUC despite class imbalance,
because class_weight="balanced" adjusted the decision boundary effectively.

## Week 3 — SHAP Explainability 

**Tools:** SHAP LinearExplainer

**Global insights:**
- tenure is the strongest predictor (SHAP = 1.12)
- InternetService_Fiber optic ranks 2nd — underestimated by correlation alone
- Contract type and MonthlyCharges round out the top 5

**Local explanation:**
- Waterfall plots show per-customer churn drivers
- High-risk customer: low tenure + fiber optic + no long-term contract + electronic check
- Low-risk customer: high tenure + two-year contract + no fiber optic

## Week 4 — Streamlit App & Deployment 


**Live App:** [Customer Churn Predictor](https://churn-predictor-iaw6tdvjpzuewk9dgofs7s.streamlit.app/)

### Features
- 🎛️ **Interactive sidebar** — input any customer's details
- 📊 **Churn probability** — real-time prediction with risk level indicator
- 🔍 **SHAP waterfall plot** — explains exactly which features drove the prediction
- 📋 **Plain English summary** — translates SHAP into business-readable insights
- 💡 **Recommended actions** — tells retention team what to do about at-risk customers

### How to run locally
```bash
streamlit run app.py
```

### App Screenshots

**High Risk Customer (95.4% churn probability)**
<img width="463" height="435" alt="Screenshot 2026-05-30 194342" src="https://github.com/user-attachments/assets/8e683b68-fc62-4bf4-b7f2-3c31293675bc" />
<img width="467" height="407" alt="Screenshot 2026-05-30 194350" src="https://github.com/user-attachments/assets/555447f7-6cbf-4c76-9f88-708fd42857fc" />
<img width="473" height="409" alt="Screenshot 2026-05-30 194356" src="https://github.com/user-attachments/assets/f218bc8f-d643-4b49-b4c2-cd8d12c3e6fe" />

> New customer on fiber optic with month-to-month contract and electronic check payment

**Low Risk Customer (1.3% churn probability)**  
<img width="958" height="438" alt="Screenshot 2026-05-30 203411" src="https://github.com/user-attachments/assets/81d36074-a305-4a28-bc6a-1e7e37f662c7" />
<img width="467" height="404" alt="Screenshot 2026-05-30 203455" src="https://github.com/user-attachments/assets/82b62788-b26b-4dea-a5ec-df404c9837a5" />
<img width="473" height="260" alt="Screenshot 2026-05-30 203502" src="https://github.com/user-attachments/assets/b5ebf0f7-b97f-4d0a-a8d7-a977b02ae3ef" />

> Long-tenure customer on two-year contract with DSL, tech support and online security

### Tech Stack
- **Frontend:** Streamlit
- **Explainability:** SHAP LinearExplainer
- **Model:** Logistic Regression (scikit-learn Pipeline)
- **Deployment:** Streamlit Cloud (free tier)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/manya-ghorpade/churn-predictor.git
cd churn-predictor

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Author

**Manya U Ghorpade** — BTech CSE (AIML)  
