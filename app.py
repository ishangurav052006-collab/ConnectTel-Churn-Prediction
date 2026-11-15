"""ConnectTel Customer Churn - Executive AI Dashboard & Retention Simulator.

An interactive Streamlit application featuring:
- Tab 1: Executive Risk Dashboard & KPI tracking
- Tab 2: Individual Customer SHAP Explainability Inspector
- Tab 3: What-If Retention Strategy & Financial ROI Simulator
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure SHAP XGBoost 3.x compatibility patch
import shap
import shap.explainers._tree as shap_tree

_orig_decode = shap_tree.decode_ubjson_buffer

def _patched_decode(fd):
    jmodel = _orig_decode(fd)
    try:
        lmp = jmodel.get("learner", {}).get("learner_model_param", {})
        if "base_score" in lmp and isinstance(lmp["base_score"], str) and lmp["base_score"].startswith("["):
            lmp["base_score"] = lmp["base_score"].strip("[]")
    except Exception:
        pass
    return jmodel

shap_tree.decode_ubjson_buffer = _patched_decode

# Set page configuration
st.set_page_config(
    page_title="ConnectTel AI Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_xgboost_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics_summary.json")


# ==============================================================================
# DATA & ARTIFACT CACHING
# ==============================================================================
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Run download_data.py first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["TotalCharges_PerTenure"] = df["TotalCharges"] / (df["tenure"] + 1e-5)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)
    df["HighRisk_Fiber_MTM"] = (
        (df["InternetService"] == "Fiber optic") & (df["Contract"] == "Month-to-month")
    ).astype(int)
    df["Tenure_Cohort"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12m", "12-24m", "24-48m", "48m+"]
    ).astype(str)
    return df


@st.cache_resource
def load_model_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Trained model not found at {MODEL_PATH}. Run train_pipeline.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


df = load_dataset()
pipeline = load_model_pipeline()
metrics_json = load_metrics()

# Precompute probabilities for the dataset
X_features = df.drop(columns=["customerID", "Churn"])
y_true = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
if "pred_prob" not in df.columns:
    df["pred_prob"] = pipeline.predict_proba(X_features)[:, 1]
    df["pred_churn"] = (df["pred_prob"] >= 0.32).astype(int) # recall-optimized threshold


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("📡 ConnectTel AI")
    st.markdown("**Enterprise Churn Retention Platform**")
    st.markdown("---")
    st.markdown("### Operational Settings")
    threshold_slider = st.slider(
        "Decision Threshold (t)",
        min_value=0.10,
        max_value=0.90,
        value=0.32,
        step=0.01,
        help="Adjusting decision threshold balances Churn Recall (Coverage) against Campaign Precision."
    )
    if threshold_slider <= 0.35:
        st.success(f"**Recall-Optimized ($F_2$) Mode**: Captures ~92.5% of churners.")
    elif threshold_slider >= 0.60:
        st.warning(f"**High-Precision Mode**: Minimizes outreach budget.")
    else:
        st.info(f"**Balanced Mode**: Standard operational setting.")
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.caption("• **Engine**: Tuned XGBoost (5-Fold CV)")
    st.caption("• **Cross-Val ROC-AUC**: 0.8493")
    st.caption("• **Minority Class Handling**: `scale_pos_weight` = 2.77")
    st.markdown("---")
    st.caption("v1.0.0 Production Release | ConnectTel Analytics")


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1f4068, #162447); padding: 22px; border-radius: 12px; color: white; margin-bottom: 25px;">
    <h1 style="margin: 0; font-size: 28px; font-weight: 700;">📡 ConnectTel Customer Churn Intelligence & Retention Suite</h1>
    <p style="margin: 6px 0 0 0; font-size: 15px; opacity: 0.9;">
        Executive analytics, individual customer explainability (XAI), and proactive lifetime-value retention modeling.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Risk Dashboard",
    "🔍 Individual Customer SHAP Inspector",
    "💡 What-If Retention Simulator"
])


# ==============================================================================
# TAB 1: EXECUTIVE RISK DASHBOARD
# ==============================================================================
with tab1:
    st.subheader("ConnectTel Macro Churn & Revenue Exposure")
    
    total_customers = len(df)
    actual_churners = int((df["Churn"] == "Yes").sum())
    churn_rate = actual_churners / total_customers
    
    # Calculate MRR at Risk
    total_mrr = df["MonthlyCharges"].sum()
    mrr_at_risk = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()
    arr_at_risk = mrr_at_risk * 12
    
    pred_churners = int((df["pred_prob"] >= threshold_slider).sum())
    pred_churn_mrr = df[df["pred_prob"] >= threshold_slider]["MonthlyCharges"].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Active Subscribers",
            value=f"{total_customers:,}",
            help="Total active and historical subscriber accounts in ConnectTel base."
        )
    with col2:
        st.metric(
            label="Historical Churn Rate",
            value=f"{churn_rate:.1%}",
            delta=f"{actual_churners:,} Churned Subscribers",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="Monthly Revenue at Risk (MRR)",
            value=f"₹{mrr_at_risk:,.0f} / mo",
            delta=f"ARR: ₹{arr_at_risk:,.0f} / yr",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label=f"Flagged by AI (t = {threshold_slider:.2f})",
            value=f"{pred_churners:,} Customers",
            delta=f"₹{pred_churn_mrr:,.0f} / mo flagged",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Visual Breakdown Section
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown("#### High-Risk Segment Exposure (Fiber Optic + Month-to-Month)")
        segment_mask = df["HighRisk_Fiber_MTM"] == 1
        seg_total = segment_mask.sum()
        seg_churn = (df.loc[segment_mask, "Churn"] == "Yes").sum()
        seg_rate = seg_churn / seg_total
        
        st.write(f"• **Segment Size**: **{seg_total:,}** customers ({seg_total/total_customers:.1%} of subscriber base)")
        st.write(f"• **Segment Churn Rate**: **{seg_rate:.1%}** (vs {df[~segment_mask]['Churn'].eq('Yes').mean():.1%} for other customers)")
        st.write(f"• **Annual Revenue at Risk in Segment**: **₹{df.loc[segment_mask & (df['Churn']=='Yes'), 'MonthlyCharges'].sum()*12:,.0f} / yr**")
        
        fig, ax = plt.subplots(figsize=(6, 3.8))
        contract_inet_churn = df.groupby(["Contract", "InternetService"])["Churn"].apply(lambda s: (s == "Yes").mean() * 100).unstack()
        contract_inet_churn.plot(kind="bar", ax=ax, edgecolor="black", width=0.7)
        ax.set_title("Churn Rate (%) across Contract & Internet Service", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)", fontweight="bold")
        ax.set_xlabel("Contract Type", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Internet Service", frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c_right:
        st.markdown("#### Benchmark Model Comparison (80/20 Test Split)")
        if metrics_json:
            comp_data = metrics_json.get("model_performance_comparison", {})
            comp_df = pd.DataFrame(comp_data).T[["roc_auc", "pr_auc", "accuracy", "recall", "precision", "f1_score"]]
            comp_df.columns = ["ROC-AUC", "PR-AUC", "Accuracy", "Recall", "Precision", "F1-Score"]
            st.dataframe(comp_df.style.format("{:.4f}").highlight_max(subset=["ROC-AUC", "PR-AUC", "Recall", "F1-Score"], color="#d4edda"), use_container_width=True)
        
        st.markdown("#### Operational Threshold Tradeoff")
        st.markdown(
            "- **Default 0.50 Threshold**: 79.68% Recall, 52.28% Precision. Captures 298 churners, misses 76.\n"
            "- **Recall-Optimized 0.32 Threshold**: **92.51% Recall**, 44.42% Precision. Captures **346 churners**, misses only **28**."
        )
        st.info("💡 **Leadership Takeaway**: Shifting to t = 0.32 enables the retention taskforce to capture 48 more churners in the test cohort alone, preserving an extra ~₹42,000 in recurring annual revenue.")


# ==============================================================================
# TAB 2: INDIVIDUAL CUSTOMER SHAP INSPECTOR
# ==============================================================================
with tab2:
    st.subheader("Customer-Level Attribution & Risk Breakdown")
    st.markdown("Select any subscriber to inspect the XGBoost predicted risk score and exact feature attribution.")
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        # Pre-filter to high-risk customers for convenient auditing
        high_risk_ids = df[df["pred_prob"] >= 0.65]["customerID"].tolist()
        sample_ids = high_risk_ids[:50] + df[df["pred_prob"] < 0.20]["customerID"].head(20).tolist()
        selected_cust_id = st.selectbox(
            "Select Customer ID (Sample includes High-Risk and Retained Profiles):",
            options=sample_ids,
            index=0
        )
    with col_sel2:
        manual_id = st.text_input("Or enter exact Customer ID:", value="")
        if manual_id and manual_id in df["customerID"].values:
            selected_cust_id = manual_id

    cust_row = df[df["customerID"] == selected_cust_id].iloc[0]
    cust_prob = float(cust_row["pred_prob"])
    
    # Top metrics card for the customer
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        if cust_prob >= 0.65:
            st.error(f"🚨 **High Risk**: {cust_prob:.1%}")
        elif cust_prob >= 0.35:
            st.warning(f"⚠️ **Elevated Risk**: {cust_prob:.1%}")
        else:
            st.success(f"✅ **Low Risk**: {cust_prob:.1%}")
    with mcol2:
        st.metric("Contract Type", cust_row["Contract"])
    with mcol3:
        st.metric("Internet Service", cust_row["InternetService"])
    with mcol4:
        st.metric("Tenure (Months)", f"{int(cust_row['tenure'])} mo")
    with mcol5:
        st.metric("Monthly Charges", f"₹{cust_row['MonthlyCharges']:.2f}")

    st.markdown("---")
    
    # Compute SHAP explanation for this individual customer
    fitted_preproc = pipeline.named_steps["preprocessor"]
    best_xgb_model = pipeline.named_steps["classifier"]
    
    raw_feature_names = fitted_preproc.get_feature_names_out()
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in raw_feature_names]
    
    cust_features = df[df["customerID"] == selected_cust_id].drop(columns=["customerID", "Churn", "pred_prob", "pred_churn"])
    cust_trans = fitted_preproc.transform(cust_features)
    cust_df = pd.DataFrame(cust_trans, columns=clean_feature_names)
    
    explainer = shap.TreeExplainer(best_xgb_model)
    single_shap = explainer(cust_df)
    
    c_sh_left, c_sh_right = st.columns([1.2, 1])
    
    with c_sh_left:
        st.markdown(f"#### SHAP Waterfall Plot for Customer `{selected_cust_id}`")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        shap.plots.waterfall(single_shap[0], max_display=10, show=False)
        plt.title(f"Attribution Breakdown - Predicted Risk: {cust_prob:.1%}", fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with c_sh_right:
        st.markdown("#### Top Factors Driving This Prediction")
        shap_vals_arr = single_shap[0].values
        feat_contrib = pd.DataFrame({
            "Feature": clean_feature_names,
            "Contribution (SHAP)": shap_vals_arr,
            "Absolute Impact": np.abs(shap_vals_arr)
        }).sort_values(by="Absolute Impact", ascending=False).head(8)
        
        for _, r in feat_contrib.iterrows():
            f_name = r["Feature"]
            val = r["Contribution (SHAP)"]
            if val > 0:
                st.markdown(f"🔴 **{f_name}**: +{val:.3f} *(Increases Churn Risk)*")
            else:
                st.markdown(f"🟢 **{f_name}**: {val:.3f} *(Protects from Churn)*")
        
        st.markdown("---")
        st.markdown("#### Recommended Retention Prescription")
        if cust_row["Contract"] == "Month-to-month" and cust_row["InternetService"] == "Fiber optic":
            st.markdown("👉 **Action**: Offer immediate 1-year contract lock with **₹15/mo discount** + free TechSupport.")
        elif cust_row["PaymentMethod"] == "Electronic check":
            st.markdown("👉 **Action**: Send automated bill credit of **₹10** to enroll in Autopay/Credit Card.")
        else:
            st.markdown("👉 **Action**: Schedule standard quarterly account health checkup.")


# ==============================================================================
# TAB 3: WHAT-IF RETENTION SIMULATOR
# ==============================================================================
with tab3:
    st.subheader("Proactive Retention Campaign & What-If Simulator")
    st.markdown(
        "Simulate contract upgrades, price discounts, and service additions to observe real-time "
        "changes in churn probability and estimated Customer Lifetime Value (LTV) preserved."
    )
    
    s_col1, s_col2 = st.columns([1.1, 1])
    
    with s_col1:
        st.markdown("### 1. Configure Customer Baseline Scenario")
        sim_contract = st.selectbox("Current Contract:", ["Month-to-month", "One year", "Two year"], index=0)
        sim_internet = st.selectbox("Internet Service:", ["Fiber optic", "DSL", "No"], index=0)
        sim_tenure = st.slider("Current Tenure (Months):", 1, 72, 6)
        sim_monthly = st.slider("Current Monthly Charges (₹):", 20.0, 120.0, 85.0, step=0.5)
        sim_partner = st.radio("Has Partner?", ["No", "Yes"], index=0, horizontal=True)
        sim_payment = st.selectbox(
            "Payment Method:",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=0
        )
        
        st.markdown("### 2. Retention Intervention Levers")
        intervene_contract = st.selectbox("Upgrade Contract to:", [sim_contract, "One year", "Two year"], index=1 if sim_contract=="Month-to-month" else 0)
        intervene_discount = st.slider("Monthly Bill Credit / Discount (₹):", 0.0, 30.0, 15.0, step=1.0)
        intervene_tech_support = st.checkbox("Bundle Free Premium TechSupport & OnlineSecurity", value=True)
        intervene_autopay = st.checkbox("Migrate Payment to Autopay (Credit Card)", value=True if sim_payment=="Electronic check" else False)

    # Compute baseline prediction
    base_dict = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": sim_partner,
        "Dependents": "No",
        "tenure": sim_tenure,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": sim_internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": sim_contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": sim_payment,
        "MonthlyCharges": sim_monthly,
        "TotalCharges": sim_monthly * sim_tenure
    }
    
    # Feature engineering for baseline
    base_df = pd.DataFrame([base_dict])
    base_df["TotalCharges_PerTenure"] = base_df["TotalCharges"] / (base_df["tenure"] + 1e-5)
    base_service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    base_df["ServiceCount"] = (base_df[base_service_cols] == "Yes").sum(axis=1)
    base_df["HighRisk_Fiber_MTM"] = ((base_df["InternetService"] == "Fiber optic") & (base_df["Contract"] == "Month-to-month")).astype(int)
    base_df["Tenure_Cohort"] = pd.cut(base_df["tenure"], bins=[-1, 12, 24, 48, np.inf], labels=["0-12m", "12-24m", "24-48m", "48m+"]).astype(str)
    
    baseline_risk = float(pipeline.predict_proba(base_df)[:, 1][0])
    
    # Build simulated intervention scenario
    sim_dict = base_dict.copy()
    sim_dict["Contract"] = intervene_contract
    sim_dict["MonthlyCharges"] = max(18.0, sim_monthly - intervene_discount)
    if intervene_tech_support:
        sim_dict["TechSupport"] = "Yes"
        sim_dict["OnlineSecurity"] = "Yes"
    if intervene_autopay:
        sim_dict["PaymentMethod"] = "Credit card (automatic)"
        
    sim_df = pd.DataFrame([sim_dict])
    sim_df["TotalCharges"] = sim_dict["MonthlyCharges"] * sim_tenure
    sim_df["TotalCharges_PerTenure"] = sim_df["TotalCharges"] / (sim_df["tenure"] + 1e-5)
    sim_df["ServiceCount"] = (sim_df[base_service_cols] == "Yes").sum(axis=1)
    sim_df["HighRisk_Fiber_MTM"] = ((sim_df["InternetService"] == "Fiber optic") & (sim_df["Contract"] == "Month-to-month")).astype(int)
    sim_df["Tenure_Cohort"] = pd.cut(sim_df["tenure"], bins=[-1, 12, 24, 48, np.inf], labels=["0-12m", "12-24m", "24-48m", "48m+"]).astype(str)
    
    simulated_risk = float(pipeline.predict_proba(sim_df)[:, 1][0])
    risk_delta = simulated_risk - baseline_risk
    
    # Financial ROI metrics
    annual_churn_cost = sim_monthly * 12
    annual_discount_cost = intervene_discount * 12
    # Expected preserved value
    expected_preservation = (baseline_risk - simulated_risk) * annual_churn_cost
    net_campaign_roi = expected_preservation - annual_discount_cost
    
    with s_col2:
        st.markdown("### 3. Real-Time Intervention Results")
        
        rc1, rc2 = st.columns(2)
        with rc1:
            st.metric(
                label="Baseline Churn Probability",
                value=f"{baseline_risk:.1%}"
            )
        with rc2:
            st.metric(
                label="Simulated Churn Probability",
                value=f"{simulated_risk:.1%}",
                delta=f"{risk_delta:.1%}",
                delta_color="inverse"
            )
            
        st.markdown("#### Projected Risk Reduction Gauge")
        st.progress(float(1.0 - simulated_risk))
        st.caption(f"Retention Probability increased from **{1.0-baseline_risk:.1%}** to **{1.0-simulated_risk:.1%}**.")
        
        st.markdown("---")
        st.markdown("#### Financial ROI Calculation (Per Account)")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.metric(
                label="Gross Annual ARR Preserved",
                value=f"₹{expected_preservation:,.0f} / yr",
                help="Expected annual recurring revenue saved based on risk reduction delta."
            )
        with f_col2:
            st.metric(
                label="Net Annual Margin Saved",
                value=f"₹{net_campaign_roi:,.0f} / yr",
                delta="Positive ROI" if net_campaign_roi > 0 else "Incentive Exceeds Benefit",
                delta_color="normal" if net_campaign_roi > 0 else "inverse"
            )
            
        st.markdown("#### Strategic Retention Assessment")
        if baseline_risk >= 0.50 and simulated_risk < 0.25:
            st.success(
                f"✅ **High-Impact Intervention**: Upgrading to **{intervene_contract}** and offering a **₹{intervene_discount:.0f}/mo** credit "
                f"reduces churn probability by **{abs(risk_delta):.1%}**, successfully securing **₹{net_campaign_roi:,.0f}** in net annual ARR."
            )
        elif simulated_risk < baseline_risk:
            st.info(f"ℹ️ **Moderate Impact**: Churn risk reduced by {abs(risk_delta):.1%}.")
        else:
            st.warning("⚠️ Intervention produces negligible impact. Consider stronger contractual lock-in.")

st.markdown("---")
st.caption("ConnectTel Customer Churn Machine Learning Platform | Production Deployment v1.0.0")

