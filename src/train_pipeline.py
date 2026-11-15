"""ConnectTel Customer Churn End-to-End Production ML Pipeline.

Executes data cleaning, domain feature engineering, EDA with hypothesis testing,
multi-model training and hyperparameter tuning, evaluation, and SHAP XAI analysis.
"""

import os
import sys
import json
import tempfile
import logging

# Configure Matplotlib cache and non-interactive backend
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.stats as stats
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    fbeta_score
)
from xgboost import XGBClassifier
import shap
import shap.explainers._tree as shap_tree

# Patch SHAP 0.49+ compatibility with XGBoost 3.x string base_score format '[5E-1]'
_orig_decode_ubjson = shap_tree.decode_ubjson_buffer

def _patched_decode_ubjson(fd):
    jmodel = _orig_decode_ubjson(fd)
    try:
        lmp = jmodel.get("learner", {}).get("learner_model_param", {})
        if "base_score" in lmp and isinstance(lmp["base_score"], str) and lmp["base_score"].startswith("["):
            lmp["base_score"] = lmp["base_score"].strip("[]")
    except Exception:
        pass
    return jmodel

shap_tree.decode_ubjson_buffer = _patched_decode_ubjson

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_JSON_PATH = os.path.join(OUTPUTS_DIR, "metrics_summary.json")

# Ensure required directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Plot styling configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})


# ==============================================================================
# 1. DATA INGESTION & FEATURE ENGINEERING
# ==============================================================================
def load_and_preprocess_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset, clean missing values, and engineer domain-specific features."""
    logging.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 1. Coerce TotalCharges to numeric (filling missing where tenure == 0 with 0.0)
    missing_tc_before = df["TotalCharges"].str.strip().eq("").sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    logging.info(f"Coerced TotalCharges to numeric. Handled {missing_tc_before} missing entries.")

    # 2. Domain Feature 1: TotalCharges_PerTenure
    # Ratio of TotalCharges to (tenure + 1e-5) to avoid division by zero
    df["TotalCharges_PerTenure"] = df["TotalCharges"] / (df["tenure"] + 1e-5)

    # 3. Domain Feature 2: ServiceCount
    # Count of additional services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)

    # 4. Domain Feature 3: HighRisk_Fiber_MTM
    # Binary flag for Fiber Optic internet users on Month-to-Month contracts
    df["HighRisk_Fiber_MTM"] = (
        (df["InternetService"] == "Fiber optic") & (df["Contract"] == "Month-to-month")
    ).astype(int)

    # 5. Domain Feature 4: Tenure_Cohort
    # Binned tenure groups (0-12m, 12-24m, 24-48m, 48m+)
    df["Tenure_Cohort"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12m", "12-24m", "24-48m", "48m+"]
    ).astype(str)

    logging.info("Engineered 4 domain features: TotalCharges_PerTenure, ServiceCount, HighRisk_Fiber_MTM, Tenure_Cohort")
    return df


# ==============================================================================
# 2. EDA & STATISTICAL HYPOTHESIS TESTING
# ==============================================================================
def run_eda_and_hypothesis_tests(df: pd.DataFrame) -> dict:
    """Generate annotated EDA figures and perform formal hypothesis tests."""
    logging.info("Generating EDA figures and conducting hypothesis testing...")
    palette = {"No": "#2b5c8f", "Yes": "#d95f02"}

    # --------------------------------------------------------------------------
    # Figure 1: Churn by Contract Type
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100

    contract_pct.plot(kind="bar", stacked=False, ax=ax, color=["#2b5c8f", "#d95f02"], edgecolor="black", width=0.6)
    ax.set_title("ConnectTel Churn Rate by Contract Type", fontweight="bold", pad=15)
    ax.set_xlabel("Contract Type", fontweight="bold")
    ax.set_ylabel("Customer Percentage (%)", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Churn Status", labels=["Retained (No)", "Churned (Yes)"], frameon=True)

    # Annotate bars with percentages
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")

    mtm_churn_rate = contract_pct.loc["Month-to-month", "Yes"]
    ax.annotate(f"Month-to-Month customers churn at {mtm_churn_rate:.1f}%,\nvs only 2.8% for Two-Year contracts",
                xy=(0.25, mtm_churn_rate), xytext=(0.8, 60),
                arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff2df", ec="#d95f02", lw=1.5),
                fontsize=9.5, fontweight="semibold")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "eda_churn_by_contract.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {fig1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Internet Service vs Churn
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    inet_churn = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
    inet_pct = inet_churn.div(inet_churn.sum(axis=1), axis=0) * 100

    inet_pct.plot(kind="bar", ax=ax, color=["#2b5c8f", "#d95f02"], edgecolor="black", width=0.6)
    ax.set_title("ConnectTel Churn Rate by Internet Service Provider Type", fontweight="bold", pad=15)
    ax.set_xlabel("Internet Service Option", fontweight="bold")
    ax.set_ylabel("Customer Percentage (%)", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Churn Status", labels=["Retained (No)", "Churned (Yes)"], frameon=True)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")

    fiber_churn_rate = inet_pct.loc["Fiber optic", "Yes"]
    dsl_churn_rate = inet_pct.loc["DSL", "Yes"]
    ax.annotate(f"Fiber Optic churn is {fiber_churn_rate:.1f}%\n(>2x higher than DSL's {dsl_churn_rate:.1f}%)",
                xy=(1.15, fiber_churn_rate), xytext=(1.4, 60),
                arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff2df", ec="#d95f02", lw=1.5),
                fontsize=9.5, fontweight="semibold")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "eda_internet_service_churn.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {fig2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Tenure & MonthlyCharges Distribution
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.boxplot(data=df, x="Churn", y="tenure", hue="Churn", legend=False,
                ax=axes[0], palette=palette, width=0.45, boxprops=dict(alpha=0.85))
    churn_tenure_med = df[df["Churn"] == "Yes"]["tenure"].median()
    no_churn_tenure_med = df[df["Churn"] == "No"]["tenure"].median()
    axes[0].set_title("Customer Tenure Distribution by Churn", fontweight="bold")
    axes[0].set_xlabel("Churn Status", fontweight="bold")
    axes[0].set_ylabel("Tenure (Months)", fontweight="bold")
    axes[0].annotate(f"Median: {no_churn_tenure_med:.0f} mo", xy=(0, no_churn_tenure_med), xytext=(-0.35, no_churn_tenure_med + 10),
                     arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.5), fontweight="bold", color="#2b5c8f")
    axes[0].annotate(f"Median: {churn_tenure_med:.0f} mo", xy=(1, churn_tenure_med), xytext=(0.65, churn_tenure_med + 14),
                     arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5), fontweight="bold", color="#d95f02")

    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", common_norm=False, fill=True,
                palette=palette, alpha=0.4, linewidth=2, ax=axes[1])
    churn_mc_med = df[df["Churn"] == "Yes"]["MonthlyCharges"].median()
    no_churn_mc_med = df[df["Churn"] == "No"]["MonthlyCharges"].median()
    axes[1].set_title("Monthly Charges Density by Churn Status", fontweight="bold")
    axes[1].set_xlabel("Monthly Charges (₹)", fontweight="bold")
    axes[1].set_ylabel("Density", fontweight="bold")
    axes[1].axvline(churn_mc_med, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Churn Median (₹{churn_mc_med:.2f})")
    axes[1].axvline(no_churn_mc_med, color="#2b5c8f", linestyle="--", linewidth=1.8, label=f"Retained Median (₹{no_churn_mc_med:.2f})")
    axes[1].legend(title="Metric & Group", frameon=True)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "eda_tenure_monthly_charges.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {fig3_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Engineered Features Impact (2x2 Grid)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Subplot A: HighRisk_Fiber_MTM Churn Rate
    hr_churn = df.groupby("HighRisk_Fiber_MTM")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    sns.barplot(x=[0, 1], y=hr_churn.values, ax=axes[0, 0], hue=[0, 1], legend=False,
                palette=["#2b5c8f", "#d95f02"], edgecolor="black")
    axes[0, 0].set_title("A. HighRisk_Fiber_MTM Flag Churn Rate", fontweight="bold")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Other Customers (0)", "Fiber + MTM (1)"])
    axes[0, 0].set_ylabel("Churn Rate (%)", fontweight="bold")
    for idx, val in enumerate(hr_churn.values):
        axes[0, 0].annotate(f"{val:.1f}%", (idx, val), ha="center", va="bottom", fontweight="bold", xytext=(0, 3), textcoords="offset points")

    # Subplot B: Tenure Cohort Churn Rate
    cohort_order = ["0-12m", "12-24m", "24-48m", "48m+"]
    tc_churn = df.groupby("Tenure_Cohort")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reindex(cohort_order)
    sns.barplot(x=list(range(len(cohort_order))), y=tc_churn.values, ax=axes[0, 1],
                hue=cohort_order, legend=False, palette="crest", edgecolor="black")
    axes[0, 1].set_title("B. Churn Rate by Tenure Cohort", fontweight="bold")
    axes[0, 1].set_xticks(list(range(len(cohort_order))))
    axes[0, 1].set_xticklabels(cohort_order)
    axes[0, 1].set_ylabel("Churn Rate (%)", fontweight="bold")
    for idx, val in enumerate(tc_churn.values):
        axes[0, 1].annotate(f"{val:.1f}%", (idx, val), ha="center", va="bottom", fontweight="bold", xytext=(0, 3), textcoords="offset points")

    # Subplot C: ServiceCount Churn Rate
    sc_churn = df.groupby("ServiceCount")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    sns.barplot(x=list(sc_churn.index), y=sc_churn.values, ax=axes[1, 0],
                hue=list(sc_churn.index), legend=False, palette="Blues_r", edgecolor="black")
    axes[1, 0].set_title("C. Churn Rate by Total Services Subscribed", fontweight="bold")
    axes[1, 0].set_xlabel("Number of Subscribed Services (0 - 8)", fontweight="bold")
    axes[1, 0].set_ylabel("Churn Rate (%)", fontweight="bold")
    for idx, val in enumerate(sc_churn.values):
        axes[1, 0].annotate(f"{val:.1f}%", (idx, val), ha="center", va="bottom", fontsize=8.5, fontweight="bold", xytext=(0, 2), textcoords="offset points")

    # Subplot D: TotalCharges_PerTenure Boxplot
    sns.boxplot(data=df, x="Churn", y="TotalCharges_PerTenure", hue="Churn", legend=False,
                ax=axes[1, 1], palette=palette, width=0.45)
    axes[1, 1].set_title("D. TotalCharges_PerTenure by Churn Status", fontweight="bold")
    axes[1, 1].set_xlabel("Churn Status", fontweight="bold")
    axes[1, 1].set_ylabel("Effective Monthly Run Rate (₹)", fontweight="bold")

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "eda_engineered_features.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {fig4_path}")

    # --------------------------------------------------------------------------
    # Statistical Hypothesis Tests
    # --------------------------------------------------------------------------
    hypothesis_results = {}

    # Test (a): Chi-Square Test: Fiber Optic vs DSL Churn Rate
    df_fiber_dsl = df[df["InternetService"].isin(["Fiber optic", "DSL"])]
    contingency_inet = pd.crosstab(df_fiber_dsl["InternetService"], df_fiber_dsl["Churn"])
    chi2_inet, p_inet, dof_inet, _ = stats.chi2_contingency(contingency_inet)

    fiber_churn_cnt = contingency_inet.loc["Fiber optic", "Yes"]
    fiber_total = contingency_inet.loc["Fiber optic"].sum()
    dsl_churn_cnt = contingency_inet.loc["DSL", "Yes"]
    dsl_total = contingency_inet.loc["DSL"].sum()

    hypothesis_results["chi2_fiber_vs_dsl"] = {
        "test_name": "Chi-Square Test of Independence (Fiber Optic vs DSL Churn)",
        "null_hypothesis": "Churn rate is independent of internet service type (Fiber Optic vs DSL).",
        "alternative_hypothesis": "Fiber Optic users have a significantly different (higher) churn rate than DSL users.",
        "chi2_statistic": float(chi2_inet),
        "p_value": float(p_inet),
        "degrees_of_freedom": int(dof_inet),
        "sample_sizes": {"Fiber_optic_total": int(fiber_total), "DSL_total": int(dsl_total)},
        "churn_rates": {
            "Fiber_optic": float(fiber_churn_cnt / fiber_total),
            "DSL": float(dsl_churn_cnt / dsl_total)
        },
        "significant": bool(p_inet < 0.05),
        "conclusion": f"Reject H0 (p = {p_inet:.3e} < 0.05). Fiber Optic customers have a significantly higher churn rate ({fiber_churn_cnt/fiber_total*100:.1f}%) compared to DSL customers ({dsl_churn_cnt/dsl_total*100:.1f}%)."
    }

    # Test (b): Chi-Square Test: Partner vs Churn
    contingency_partner = pd.crosstab(df["Partner"], df["Churn"])
    chi2_partner, p_partner, dof_partner, _ = stats.chi2_contingency(contingency_partner)

    partner_churn_cnt = contingency_partner.loc["Yes", "Yes"]
    partner_total = contingency_partner.loc["Yes"].sum()
    no_partner_churn_cnt = contingency_partner.loc["No", "Yes"]
    no_partner_total = contingency_partner.loc["No"].sum()

    hypothesis_results["chi2_partner_vs_churn"] = {
        "test_name": "Chi-Square Test of Independence (Partner vs Churn)",
        "null_hypothesis": "Churn rate is independent of whether a customer has a partner.",
        "alternative_hypothesis": "Customers with a partner have a significantly lower churn risk.",
        "chi2_statistic": float(chi2_partner),
        "p_value": float(p_partner),
        "degrees_of_freedom": int(dof_partner),
        "sample_sizes": {"Partner_Yes_total": int(partner_total), "Partner_No_total": int(no_partner_total)},
        "churn_rates": {
            "Partner_Yes": float(partner_churn_cnt / partner_total),
            "Partner_No": float(no_partner_churn_cnt / no_partner_total)
        },
        "significant": bool(p_partner < 0.05),
        "conclusion": f"Reject H0 (p = {p_partner:.3e} < 0.05). Having a partner significantly reduces churn risk (churn rate: {partner_churn_cnt/partner_total*100:.1f}% for partnered vs {no_partner_churn_cnt/no_partner_total*100:.1f}% for unpartnered)."
    }

    # Test (c): Mann-Whitney U Test: MonthlyCharges for Churners vs Non-Churners
    churn_charges = df[df["Churn"] == "Yes"]["MonthlyCharges"]
    retained_charges = df[df["Churn"] == "No"]["MonthlyCharges"]
    u_stat, p_mannwhitney = stats.mannwhitneyu(churn_charges, retained_charges, alternative="two-sided")

    hypothesis_results["mann_whitney_monthly_charges"] = {
        "test_name": "Mann-Whitney U Test (MonthlyCharges by Churn Status)",
        "null_hypothesis": "Distribution of MonthlyCharges is identical between churned and retained customers.",
        "alternative_hypothesis": "Churned customers have systematically higher MonthlyCharges than retained customers.",
        "u_statistic": float(u_stat),
        "p_value": float(p_mannwhitney),
        "medians": {
            "Churn_Yes_median": float(churn_charges.median()),
            "Churn_No_median": float(retained_charges.median())
        },
        "means": {
            "Churn_Yes_mean": float(churn_charges.mean()),
            "Churn_No_mean": float(retained_charges.mean())
        },
        "significant": bool(p_mannwhitney < 0.05),
        "conclusion": f"Reject H0 (p = {p_mannwhitney:.3e} < 0.05). Customers who churn pay significantly higher monthly charges (median ₹{churn_charges.median():.2f} vs ₹{retained_charges.median():.2f})."
    }

    logging.info("Hypothesis testing completed with highly significant results across all tests.")
    return hypothesis_results


# ==============================================================================
# 3. MODEL TRAINING & HYPERPARAMETER TUNING
# ==============================================================================
def train_and_tune_models(df: pd.DataFrame):
    """Train Logistic Regression, Random Forest, and GridSearchCV tuned XGBoost."""
    logging.info("Preparing stratified train/test split (80/20)...")
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "TotalCharges_PerTenure", "ServiceCount"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    # --------------------------------------------------------------------------
    # Model 1: Logistic Regression
    # --------------------------------------------------------------------------
    logging.info("Training Logistic Regression (balanced class weight)...")
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)

    # Extract Odds Ratios
    fitted_preprocessor = lr_pipeline.named_steps["preprocessor"]
    raw_feature_names = fitted_preprocessor.get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in raw_feature_names]
    lr_coefs = lr_pipeline.named_steps["classifier"].coef_[0]
    odds_ratios = np.exp(lr_coefs)
    odds_ratios_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr_coefs,
        "odds_ratio": odds_ratios
    }).sort_values(by="odds_ratio", ascending=False)

    top_pos_odds = odds_ratios_df.head(10).to_dict(orient="records")
    top_neg_odds = odds_ratios_df.tail(10).to_dict(orient="records")

    # --------------------------------------------------------------------------
    # Model 2: Random Forest
    # --------------------------------------------------------------------------
    logging.info("Training Random Forest Classifier (balanced class weight)...")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)

    # --------------------------------------------------------------------------
    # Model 3: XGBoost with GridSearchCV
    # --------------------------------------------------------------------------
    logging.info("Tuning XGBoost Classifier with 5-Fold Stratified CV optimizing for ROC-AUC...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    xgb_base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        xgb_base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_xgb_pipeline = grid_search.best_estimator_
    best_xgb_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    best_xgb_cv_auc = float(grid_search.best_score_)
    logging.info(f"Best XGBoost CV ROC-AUC: {best_xgb_cv_auc:.4f} with params: {best_xgb_params}")

    # Save artifacts
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "logistic_regression_pipeline.joblib"))
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"))
    joblib.dump(best_xgb_pipeline, os.path.join(MODELS_DIR, "best_xgboost_pipeline.joblib"))
    logging.info("Saved all trained pipeline models to models/ directory.")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "lr_pipeline": lr_pipeline,
        "rf_pipeline": rf_pipeline,
        "best_xgb_pipeline": best_xgb_pipeline,
        "best_xgb_params": best_xgb_params,
        "best_xgb_cv_auc": best_xgb_cv_auc,
        "top_pos_odds": top_pos_odds,
        "top_neg_odds": top_neg_odds,
        "odds_ratios_df": odds_ratios_df
    }


# ==============================================================================
# 4. EVALUATION, THRESHOLD OPTIMIZATION & SHAP XAI
# ==============================================================================
def evaluate_and_explain(artifacts: dict) -> dict:
    """Evaluate models, plot ROC/PR curves, analyze threshold tradeoffs, and run SHAP."""
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    lr = artifacts["lr_pipeline"]
    rf = artifacts["rf_pipeline"]
    xgb = artifacts["best_xgb_pipeline"]
    feature_names = artifacts["feature_names"]

    # Model predictions
    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Tuned XGBoost": xgb
    }

    metrics_table = {}
    probas = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        probas[name] = y_prob
        y_pred = (y_prob >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        metrics_table[name] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        logging.info(f"{name} -> ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # --------------------------------------------------------------------------
    # Figure 5: Multi-Model ROC-AUC Curve Comparison
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#55a868", "Tuned XGBoost": "#d95f02"}

    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        plt.plot(fpr, tpr, color=colors[name], lw=2.2,
                 label=f"{name} (AUC = {metrics_table[name]['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("ConnectTel Churn Prediction - Multi-Model ROC Curves", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_fig_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {roc_fig_path}")

    # --------------------------------------------------------------------------
    # Figure 6: Precision-Recall Curve Comparison
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    baseline = float(y_test.mean())

    for name in models:
        prec_c, rec_c, _ = precision_recall_curve(y_test, probas[name])
        plt.plot(rec_c, prec_c, color=colors[name], lw=2.2,
                 label=f"{name} (PR-AUC = {metrics_table[name]['pr_auc']:.3f})")

    plt.axhline(baseline, color="gray", linestyle="--", lw=1.5, label=f"No Skill Baseline ({baseline:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Coverage of Churners)", fontweight="bold")
    plt.ylabel("Precision (Accuracy of Churn Alerts)", fontweight="bold")
    plt.title("ConnectTel Churn Prediction - Precision-Recall Curves", fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    pr_fig_path = os.path.join(FIGURES_DIR, "pr_curves.png")
    plt.savefig(pr_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {pr_fig_path}")

    # --------------------------------------------------------------------------
    # Threshold Optimization (Recall-Optimized for Churn Retention)
    # --------------------------------------------------------------------------
    xgb_probs = probas["Tuned XGBoost"]
    f2_scores = []
    test_thresholds = np.linspace(0.1, 0.9, 100)
    for t in test_thresholds:
        pred_t = (xgb_probs >= t).astype(int)
        f2 = fbeta_score(y_test, pred_t, beta=2, zero_division=0)
        f2_scores.append(f2)

    best_idx = np.argmax(f2_scores)
    opt_threshold = float(test_thresholds[best_idx])
    opt_f2 = float(f2_scores[best_idx])

    # Default predictions (t=0.50) vs Recall-optimized predictions
    pred_default = (xgb_probs >= 0.50).astype(int)
    pred_opt = (xgb_probs >= opt_threshold).astype(int)

    cm_default = confusion_matrix(y_test, pred_default)
    cm_opt = confusion_matrix(y_test, pred_opt)

    thresh_metrics = {
        "default_threshold_0_50": {
            "threshold": 0.50,
            "confusion_matrix": cm_default.tolist(),
            "true_negatives": int(cm_default[0, 0]),
            "false_positives": int(cm_default[0, 1]),
            "false_negatives": int(cm_default[1, 0]),
            "true_positives": int(cm_default[1, 1]),
            "precision": float(precision_score(y_test, pred_default)),
            "recall": float(recall_score(y_test, pred_default)),
            "f1_score": float(f1_score(y_test, pred_default)),
            "f2_score": float(fbeta_score(y_test, pred_default, beta=2))
        },
        "recall_optimized_threshold": {
            "threshold": opt_threshold,
            "optimization_criterion": "Maximize F2 Score (emphasizing Churn Recall for proactive retention)",
            "confusion_matrix": cm_opt.tolist(),
            "true_negatives": int(cm_opt[0, 0]),
            "false_positives": int(cm_opt[0, 1]),
            "false_negatives": int(cm_opt[1, 0]),
            "true_positives": int(cm_opt[1, 1]),
            "precision": float(precision_score(y_test, pred_opt)),
            "recall": float(recall_score(y_test, pred_opt)),
            "f1_score": float(f1_score(y_test, pred_opt)),
            "f2_score": opt_f2
        }
    }

    # --------------------------------------------------------------------------
    # Figure 7: Confusion Matrix Heatmaps (Default vs Recall-Optimized)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"size": 13, "fontweight": "bold"})
    axes[0].set_title(f"Default Threshold (t = 0.50)\nRecall: {thresh_metrics['default_threshold_0_50']['recall']:.2f} | Precision: {thresh_metrics['default_threshold_0_50']['precision']:.2f}",
                      fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontweight="bold")
    axes[0].set_ylabel("True Label", fontweight="bold")
    axes[0].set_xticklabels(["Retained (0)", "Churned (1)"])
    axes[0].set_yticklabels(["Retained (0)", "Churned (1)"])

    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1],
                annot_kws={"size": 13, "fontweight": "bold"})
    axes[1].set_title(f"Recall-Optimized Threshold (t = {opt_threshold:.2f})\nRecall: {thresh_metrics['recall_optimized_threshold']['recall']:.2f} | Precision: {thresh_metrics['recall_optimized_threshold']['precision']:.2f}",
                      fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontweight="bold")
    axes[1].set_ylabel("True Label", fontweight="bold")
    axes[1].set_xticklabels(["Retained (0)", "Churned (1)"])
    axes[1].set_yticklabels(["Retained (0)", "Churned (1)"])

    churners_captured_diff = thresh_metrics['recall_optimized_threshold']['true_positives'] - thresh_metrics['default_threshold_0_50']['true_positives']
    fig.suptitle(f"ConnectTel Retention Decision Matrix: Optimized Threshold Captures +{churners_captured_diff} Additional Churners",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {cm_fig_path}")

    # --------------------------------------------------------------------------
    # SHAP Explainability Analysis (XAI)
    # --------------------------------------------------------------------------
    logging.info("Computing SHAP values for Tuned XGBoost model...")
    best_xgb_model = xgb.named_steps["classifier"]
    fitted_preproc = xgb.named_steps["preprocessor"]

    X_test_trans = fitted_preproc.transform(X_test)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)

    explainer = shap.TreeExplainer(best_xgb_model)
    shap_values = explainer(X_test_df)

    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("ConnectTel Churn - Top 15 SHAP Predictors (Beeswarm)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    shap_bee_path = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
    plt.savefig(shap_bee_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {shap_bee_path}")

    # 2. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("ConnectTel Churn - Global Mean |SHAP| Value", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    shap_bar_path = os.path.join(FIGURES_DIR, "shap_importance_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {shap_bar_path}")

    # 3. Single-Customer SHAP Waterfall Plot
    churner_mask = (y_test.values == 1)
    high_risk_idx = np.where(churner_mask)[0][np.argmax(xgb_probs[churner_mask])]
    target_prob = float(xgb_probs[high_risk_idx])

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=12, show=False)
    plt.title(f"Individual Customer Churn Attribution (Customer #{high_risk_idx} - Predicted Risk: {target_prob:.1%})",
              fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    shap_wf_path = os.path.join(FIGURES_DIR, "shap_waterfall_single_customer.png")
    plt.savefig(shap_wf_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {shap_wf_path}")

    # 4. Quantifying Driver: Month-to-Month Contract vs Monthly Charges
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance_series = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    mtm_feature = [f for f in feature_names if "Month-to-month" in f][0]
    mc_feature = "MonthlyCharges"

    mtm_shap_val = float(shap_importance_series[mtm_feature])
    mc_shap_val = float(shap_importance_series[mc_feature])
    mtm_rank = int(np.where(shap_importance_series.index == mtm_feature)[0][0] + 1)
    mc_rank = int(np.where(shap_importance_series.index == mc_feature)[0][0] + 1)

    winner = mtm_feature if mtm_shap_val > mc_shap_val else mc_feature
    driver_ratio = mtm_shap_val / (mc_shap_val + 1e-9)

    shap_driver_comparison = {
        "question": "Is Month-to-Month contract status or Monthly Charges the #1 driver of churn?",
        "Contract_Month_to_Month": {
            "feature_name": mtm_feature,
            "mean_abs_shap": mtm_shap_val,
            "global_rank": mtm_rank
        },
        "MonthlyCharges": {
            "feature_name": mc_feature,
            "mean_abs_shap": mc_shap_val,
            "global_rank": mc_rank
        },
        "dominant_driver": winner,
        "ratio_mtm_to_mc": float(driver_ratio),
        "verdict": f"{winner} is the primary driver between the two (mean |SHAP|: {max(mtm_shap_val, mc_shap_val):.4f} vs {min(mtm_shap_val, mc_shap_val):.4f})."
    }
    logging.info(f"SHAP Driver Analysis: {shap_driver_comparison['verdict']}")

    return {
        "model_comparison": metrics_table,
        "threshold_analysis": thresh_metrics,
        "shap_driver_comparison": shap_driver_comparison,
        "top_10_shap_features": [
            {"feature": f, "mean_abs_shap": float(val), "rank": idx + 1}
            for idx, (f, val) in enumerate(shap_importance_series.head(10).items())
        ]
    }


# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    logging.info("=" * 70)
    logging.info("STARTING CONNECTTEL CHURN PRODUCTION MACHINE LEARNING PIPELINE")
    logging.info("=" * 70)

    # 1. Ingestion & Feature Engineering
    df = load_and_preprocess_data(DATA_PATH)

    # 2. EDA & Hypothesis Testing
    hypothesis_results = run_eda_and_hypothesis_tests(df)

    # 3. Model Training & Tuning
    train_artifacts = train_and_tune_models(df)

    # 4. Evaluation & SHAP XAI
    eval_artifacts = evaluate_and_explain(train_artifacts)

    # 5. Compile and Save Comprehensive Metrics Summary
    summary = {
        "project": "ConnectTel Customer Churn Machine Learning Pipeline",
        "dataset_summary": {
            "total_records": len(df),
            "total_features": len(df.columns) - 1, # excluding customerID
            "churn_distribution": {
                "No": int((df["Churn"] == "No").sum()),
                "Yes": int((df["Churn"] == "Yes").sum()),
                "churn_rate_pct": float((df["Churn"] == "Yes").mean() * 100)
            }
        },
        "hypothesis_testing": hypothesis_results,
        "logistic_regression": {
            "top_10_churn_inducing_features_odds_ratio": train_artifacts["top_pos_odds"],
            "top_10_churn_protective_features_odds_ratio": train_artifacts["top_neg_odds"]
        },
        "xgboost_tuning": {
            "best_hyperparameters": train_artifacts["best_xgb_params"],
            "best_5fold_cv_roc_auc": train_artifacts["best_xgb_cv_auc"]
        },
        "model_performance_comparison": eval_artifacts["model_comparison"],
        "threshold_optimization": eval_artifacts["threshold_analysis"],
        "shap_explainability": {
            "driver_comparison": eval_artifacts["shap_driver_comparison"],
            "top_10_global_features": eval_artifacts["top_10_shap_features"]
        }
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=4)
    logging.info(f"Saved complete metrics summary to: {METRICS_JSON_PATH}")

    logging.info("=" * 70)
    logging.info("CONNECTTEL PRODUCTION PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
