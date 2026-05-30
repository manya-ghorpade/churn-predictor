import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
import joblib

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ── Load model ──────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("data/churn_model.pkl")

pipeline = load_model()

# ── Title ───────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.markdown("Enter customer details in the sidebar to predict churn probability and understand the key drivers.")
st.divider()

st.sidebar.header("Customer Details")

# Numerical inputs
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

# Categorical inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# ── Build input dataframe ────────────────────────────────
input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# ── Predict ──────────────────────────────────────────────
churn_prob = pipeline.predict_proba(input_df)[0][1]
churn_label = "⚠️ Likely to Churn" if churn_prob >= 0.5 else "✅ Likely to Stay"

# ── Layout: two columns ──────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    st.metric("Churn Probability", f"{churn_prob:.1%}")

    if churn_prob >= 0.5:
        st.error(churn_label)
    else:
        st.success(churn_label)

    st.markdown("**Risk Level:**")
    if churn_prob < 0.3:
        st.progress(churn_prob, text="Low Risk")
    elif churn_prob < 0.6:
        st.progress(churn_prob, text="Medium Risk")
    else:
        st.progress(churn_prob, text="High Risk")

with col2:
    st.subheader("Why this prediction? (SHAP Explanation)")

    # SHAP
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.default"] = "regular"

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [col for col in input_df.columns if col not in num_cols]

    # Load training data as background for SHAP
    df_full = pd.read_csv("data/telco_churn_clean.csv").drop(columns=["Churn"])
    X_background = preprocessor.transform(df_full)
    X_input = preprocessor.transform(input_df)

    cat_feature_names = preprocessor\
        .named_transformers_["cat"]\
        .get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    explainer = shap.LinearExplainer(
        model, X_background,
        feature_names=all_feature_names
    )
    shap_values = explainer(X_input)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    st.pyplot(plt.gcf())
    plt.close("all")
    # Plain English explanation
    st.markdown("---")
    st.subheader("📋 What this means in plain English")

    # Get top churn-driving features
    shap_df = pd.DataFrame({
        "feature": all_feature_names,
        "shap_value": shap_values[0].values
    }).sort_values("shap_value", ascending=False)

    top_risks = shap_df[shap_df["shap_value"] > 0.1].head(3)
    top_savers = shap_df[shap_df["shap_value"] < -0.1].tail(3)
    # Clean feature name mapping
    def get_feature_label(feature, shap_val):
        churn_labels = {
            "tenure": "Short customer tenure (new customer)",
            "Contract_Two year": "No long-term contract",
            "Contract_One year": "No long-term contract",
            "PaymentMethod_Electronic check": "Electronic check payment method",
            "InternetService_Fiber optic": "Fiber optic internet service",
            "MonthlyCharges": "High monthly charges",
            "TotalCharges": "Low total charges (new customer)",
            "TechSupport_Yes": "No tech support",
            "OnlineSecurity_Yes": "No online security",
            "SeniorCitizen_1": "Senior citizen customer",
            "PaperlessBilling_Yes": "Paperless billing enabled",
        }
        stay_labels = {
            "tenure": "Long customer tenure (loyal customer)",
            "Contract_Two year": "Two-year contract commitment",
            "InternetService_Fiber optic": "No fiber optic (lower risk service)",
            "OnlineSecurity_Yes": "Has online security",
            "TechSupport_Yes": "Has tech support",
            "Dependents_Yes": "Has dependents (stable household)",
            "MonthlyCharges": "Lower monthly charges",
        }
        if shap_val > 0:
            return churn_labels.get(feature, feature)
        else:
            return stay_labels.get(feature, feature)

    if churn_prob >= 0.5:
        st.error("🚨 This customer is at high risk of leaving. Here's why:")
        for _, row in top_risks.iterrows():
            label = get_feature_label(row['feature'], row['shap_value'])
            st.markdown(f"- **{label}**")

        st.info("💡 Recommended Actions:")
        if any("Contract" in f for f in top_risks["feature"].values):
            st.markdown("- 📄 Offer a discounted **long-term contract**")
        if any("TechSupport" in f for f in top_risks["feature"].values):
            st.markdown("- 🛠️ Offer free **Tech Support** for 3 months")
        if any("tenure" in f for f in top_risks["feature"].values):
            st.markdown("- 🎁 Offer a **new customer loyalty bonus**")
        if any("Electronic check" in f for f in top_risks["feature"].values):
            st.markdown("- 💳 Encourage switch to **automatic payment**")
    else:
        st.success("✅ This customer is likely to stay. Key loyalty factors:")
        for _, row in top_savers.iterrows():
            label = get_feature_label(row['feature'], row['shap_value'])
            st.markdown(f"- **{label}**")
   
