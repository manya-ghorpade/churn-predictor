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

## Week 2 — ML Models *(in progress)*

## Week 3 — SHAP Explainability *(coming soon)*

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
