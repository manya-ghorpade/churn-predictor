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
│   └── EDA.ipynb                               # Week 1 — EDA
├── src/                                        # ML pipeline (Week 2)
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

## Week 4 — Streamlit App & Deployment *(coming soon)*

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
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
