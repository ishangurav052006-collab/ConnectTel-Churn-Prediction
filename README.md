# 📡 ConnectTel Customer Churn Prediction & Retention Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.49.1-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-ff4b4b.svg)](https://streamlit.io/)

A production-ready machine learning, statistical hypothesis testing, explainable AI (XAI), and proactive retention platform developed for **ConnectTel Telecommunications**.

---

## 📑 Table of Contents
1. [Executive Overview & Problem Statement](#-executive-overview--problem-statement)
2. [Project Architecture](#-project-architecture)
3. [Statistical Hypothesis Testing Findings](#-statistical-hypothesis-testing-findings)
4. [Domain Feature Engineering](#-domain-feature-engineering)
5. [Model Training & Cross-Validation Benchmarks](#-model-training--cross-validation-benchmarks)
6. [Operational Threshold Optimization](#-operational-threshold-optimization)
7. [Explainable AI (SHAP Analysis)](#-explainable-ai-shap-analysis)
8. [Actionable Retention Strategies & ROI Model](#-actionable-retention-strategies--roi-model)
9. [Installation & Execution Guide](#-installation--execution-guide)

---

## 🎯 Executive Overview & Problem Statement

ConnectTel faces an overall historical subscriber churn rate of **26.54%** (1,869 churners out of 7,043 total customers). In the telecommunications sector, Customer Acquisition Cost (CAC) exceeds retention cost by **5x to 10x**.

### The Economic Asymmetry of Classification Errors:
* **False Negative (FN) - Cost of Missing a Churner:** The customer defects to a competitor. ConnectTel permanently loses recurring subscription revenue ($ARPU \approx ₹74.44$/mo or **₹893.30 annually**), sacrifices Customer Lifetime Value (LTV), and incurs ₹300 - ₹500 in replacement acquisition marketing.
* **False Positive (FP) - Cost of Unnecessary Outreach:** ConnectTel offers a retention discount or complimentary feature to a subscriber who would not have churned. The expense is strictly limited to the margin impact of the incentive (₹10 - ₹15/mo).

$$\text{Error Cost Asymmetry Ratio} \approx \frac{\text{Cost}(FN)}{\text{Cost}(FP)} \approx 6:1 \text{ to } 10:1$$

### Why Raw Accuracy is Misleading:
On ConnectTel's imbalanced dataset (73.46% retained vs 26.54% churned), a naive majority-class model achieves **73.46% Accuracy** while capturing **0% of churners** (₹0 saved). Consequently, this platform optimizes for **ROC-AUC** (discriminative ranking power) and **Recall** ($F_2$-Score, heavily penalizing false negatives for outreach execution).

---

## 🏗 Project Architecture

```
ConnectTel-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv          # Raw IBM Telco customer dataset (7,043 x 21)
│
├── src/
│   ├── __init__.py                       # Package definition
│   ├── download_data.py                  # Automated dataset ingestion script
│   ├── verify_data.py                    # Data integrity and health check auditor
│   ├── train_pipeline.py                 # End-to-end training, EDA, tuning & SHAP pipeline
│   ├── build_notebook.py                 # Programmatic nbformat Jupyter notebook builder
│   └── create_submission_zip.py          # Portable ZIP packaging utility
│
├── models/
│   ├── logistic_regression_pipeline.joblib  # Trained Logistic Regression pipeline
│   ├── random_forest_pipeline.joblib        # Trained Random Forest pipeline
│   └── best_xgboost_pipeline.joblib         # Tuned XGBoost production pipeline
│
├── notebooks/
│   └── ConnectTel_Churn_Analysis.ipynb   # Comprehensive executive Jupyter notebook
│
├── outputs/
│   ├── figures/                          # 10 High-resolution publication plots (300 DPI)
│   │   ├── eda_churn_by_contract.png
│   │   ├── eda_internet_service_churn.png
│   │   ├── eda_tenure_monthly_charges.png
│   │   ├── eda_engineered_features.png
│   │   ├── roc_curves.png
│   │   ├── pr_curves.png
│   │   ├── confusion_matrices.png
│   │   ├── shap_summary_beeswarm.png
│   │   ├── shap_importance_bar.png
│   │   └── shap_waterfall_single_customer.png
│   └── metrics_summary.json              # Full metrics, odds ratios, and test statistics
│
├── app.py                                # Interactive 3-Tab Streamlit web application
├── ConnectTel_Churn_Analysis.ipynb       # Executed root copy of the analysis notebook
├── requirements.txt                      # Production environment dependencies
├── .gitignore                            # Environment and artifact exclusions
└── README.md                             # Comprehensive technical & executive documentation
```

---

## 🔬 Statistical Hypothesis Testing Findings

Formal hypothesis testing was conducted to validate observable operational patterns before modeling:

| Formal Hypothesis Test | Null Hypothesis ($H_0$) | Test Statistic | Exact $p$-value | Significance | Business Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Chi-Square: Fiber Optic vs DSL Churn** | Churn rate is independent of internet infrastructure | $\chi^2 = 328.263$ | $\mathbf{2.30 \times 10^{-73}}$ | $p < 0.001$ | **Reject $H_0$**: Fiber Optic customers churn at **41.9%**, more than double DSL customers (**19.0%**). |
| **Chi-Square: Partner Status vs Churn** | Churn rate is independent of household partner status | $\chi^2 = 158.733$ | $\mathbf{2.14 \times 10^{-36}}$ | $p < 0.001$ | **Reject $H_0$**: Having a partner significantly lowers churn risk from **33.0%** (single) down to **19.7%** (partnered). |
| **Mann-Whitney U: Monthly Charges** | MonthlyCharges distributions are identical | $U = 6,003,125.5$ | $\mathbf{3.31 \times 10^{-54}}$ | $p < 0.001$ | **Reject $H_0$**: Churners pay significantly higher monthly rates (Median: **₹79.65** vs **₹64.43**). |

---

## ⚙️ Domain Feature Engineering

Four targeted domain features were engineered to capture customer behavioral signals:
1. `TotalCharges_PerTenure`: Ratio of `TotalCharges / (tenure + 1e-5)` reflecting the effective monthly billing run rate.
2. `ServiceCount`: Cumulative count (0 to 8) of supplementary services adopted across Phone, Multiple Lines, Online Security, Online Backup, Device Protection, TechSupport, Streaming TV, and Streaming Movies.
3. `HighRisk_Fiber_MTM`: Binary interaction identifying the most vulnerable cohort (Fiber Optic users on Month-to-Month contracts), which exhibits an alarming **54.6% churn rate**.
4. `Tenure_Cohort`: Lifecycle binned segments (`0-12m`, `12-24m`, `24-48m`, `48m+`).

---

## 📊 Model Training & Cross-Validation Benchmarks

Models were evaluated on an **80/20 stratified test split** using a production `ColumnTransformer` (`StandardScaler` + `OneHotEncoder(handle_unknown='ignore')`):

| Model Architecture | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test $F_2$-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost** | **0.8493** | **0.8458** | **0.6590** | 75.30% | 52.28% | **79.68%** | **0.6314** | **0.7212** |
| **Logistic Regression (Balanced)** | 0.8415 | 0.8419 | 0.6359 | 73.31% | 49.83% | 79.14% | 0.6116 | 0.7077 |
| **Random Forest (Balanced)** | 0.8241 | 0.8223 | 0.6083 | **78.50%** | **61.95%** | 49.20% | 0.5484 | 0.5126 |

* **Optimal XGBoost Hyperparameters**: `learning_rate`: 0.05, `max_depth`: 3, `n_estimators`: 100, `subsample`: 0.8, `scale_pos_weight`: 2.77.
* **Top Logistic Regression Odds Ratios**:
  - Churn-inducing: `Contract_Month-to-month` ($OR = 2.04$), `InternetService_Fiber optic` ($OR = 2.02$).
  - Protective: `tenure` ($OR = 0.36$), `Contract_Two year` ($OR = 0.43$), `InternetService_DSL` ($OR = 0.53$).

---

## ⚖️ Operational Threshold Optimization

To minimize the high cost of false negatives, the operational decision threshold was calibrated on $F_2$-score (giving Recall double the weight of Precision):

```
+-----------------------------------------------------------------------------------------+
| STRATEGY                 | THRESHOLD | RECALL  | PRECISION | TRUE POSITIVES | MISSED (FN)  |
+--------------------------+-----------+---------+-----------+----------------+--------------+
| Default Setting          |  t = 0.50 |  79.68% |    52.28% |  298 Churners  |  76 Missed   |
| Recall-Optimized Setting |  t = 0.32 |  92.51% |    44.42% |  346 Churners  |  28 Missed   |
+-----------------------------------------------------------------------------------------+
| OPERATIONAL BENEFIT: Captures +48 ADDITIONAL CHURNERS (Cutting missed churners by 63%)   |
+-----------------------------------------------------------------------------------------+
```

---

## 🔍 Explainable AI (SHAP Analysis)

Shapley Additive exPlanations (`shap.TreeExplainer`) were computed on the tuned XGBoost model to answer ConnectTel's primary strategic question:  
**Is Month-to-Month Contract status or high Monthly Charges the #1 driver of customer churn?**

| Candidate Strategic Driver | Feature Name | Mean \|SHAP\| Value | Global Model Rank | Relative Impact Ratio |
| :--- | :--- | :---: | :---: | :---: |
| **Month-to-Month Contract Status** | `Contract_Month-to-month` | **0.5862** | **Rank #1 (Dominant)** | **6.15x** |
| **Monthly Subscription Charges** | `MonthlyCharges` | **0.0953** | **Rank #11** | 1.00x (Baseline) |

* **Definitive Conclusion**: Contract flexibility is **over 6.15 times more decisive** than monthly charges. Subscribers do not churn simply because pricing is elevated; they churn when elevated pricing coincides with **zero switching friction** (a Month-to-Month contract).

---

## 💼 Actionable Retention Strategies & ROI Model

### 1. Targeted 1-Year Contract Migration Campaign
* **Target Cohort**: Fiber Optic subscribers on Month-to-Month contracts (2,128 customers, baseline churn: **54.6%**).
* **Intervention**: Offer a **₹15/month billing credit** (or complimentary high-speed router + TechSupport) in exchange for signing a 1-year contract commitment.

### Financial ROI Projection:

| Strategic Parameter | Financial Value |
| :--- | :--- |
| **Segment Population (Fiber + MTM)** | 2,128 subscribers |
| **Baseline Annual Churners** | 1,162 subscribers / year |
| **Annual Recurring Revenue (ARR) at Risk** | **₹1,220,100 / year** (₹87.50/mo ARPU) |
| **Campaign Conversion Rate (Assumed)** | 30.0% (638 subscribers enroll) |
| **Churn Rate Reduction Among Converts** | Drops from 54.6% to 11.2% (Contract baseline) |
| **Net Retained Subscribers** | **277 subscribers saved / year** |
| **Gross Preserved ARR** | **₹290,850 / year** |
| **Annual Retention Discount Cost** | ₹114,840 / year ($638 \times ₹180$/yr credit) |
| **Net Annual Bottom-Line Benefit (Year 1)** | **$\mathbf{+₹176,010}$ / year** |
| **Campaign Return on Investment (ROI)** | **$\mathbf{253\%}$ (2.53x Financial Return)** |
| **3-Year Customer LTV Preserved** | **$\mathbf{+₹580,000+}$** |

### Additional Operational Tactics:
* **TechSupport / Security Onboarding**: Automatically include 90 days of free premium TechSupport with Fiber installations (EDA showed lack of TechSupport correlates with a 41.7% churn rate).
* **Autopay Migration**: Provide a one-time ₹10 bill credit to transition Electronic Check users (45.3% churn) to automated credit card/ACH bank transfer (15-18% churn).

---

## 🚀 Installation & Execution Guide

### 1. Environment Setup
```bash
# Clone or navigate to the workspace
cd ConnectTel-Churn-Prediction

# Install all required packages
pip install -r requirements.txt
```

### 2. Download Data
```bash
python src/download_data.py
```

### 3. Run End-to-End Training & Pipeline
```bash
python src/train_pipeline.py
```
*Outputs: Generates all 10 high-resolution plots in `outputs/figures/`, serialized pipelines in `models/`, and metrics in `outputs/metrics_summary.json`.*

### 4. Build & Execute Jupyter Notebook
```bash
# Build the executive notebook
python src/build_notebook.py

# Execute notebook end-to-end with inline outputs
jupyter nbconvert --to notebook --execute --inplace notebooks/ConnectTel_Churn_Analysis.ipynb
```

### 5. Launch Interactive Streamlit App
```bash
streamlit run app.py
```
*Access the 3-tab web dashboard locally at `http://localhost:8501`:*
- **Tab 1**: Executive Risk Dashboard & KPI tracking.
- **Tab 2**: Individual Customer SHAP Inspector.
- **Tab 3**: What-If Retention Strategy & Financial ROI Simulator.

### 6. Create Submission ZIP Archive
```bash
python src/create_submission_zip.py
```
*Packages the entire project into a clean, submission-ready `ConnectTel_Churn_Submission.zip`.*

---

## 👥 Authors & License
- **Project**: ConnectTel Customer Churn Machine Learning Platform
- **Lead Data Scientist**: Ishan Gurav
- **License**: MIT
