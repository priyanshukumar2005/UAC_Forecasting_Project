# 👶 Predictive Forecasting of UAC Care Load & Placement Demand

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20Scikit--Learn%20%7C%20Statsmodels-orange.svg)](https://scikit-learn.org/)
[![Maintenance](https://img.shields.io/badge/status-active%20%2F%20production%20ready-brightgreen.svg)]()

> An end-to-end predictive intelligence system designed to forecast Unaccompanied Children (UAC) care load, model upstream custody flow dynamics, and deliver early warnings for health operations and shelter capacity planning.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Usage Instructions](#-usage-instructions)
- [Dataset Description](#-dataset-description)
- [Model Architecture & Benchmarking](#-model-architecture--benchmarking)
- [KPIs & Risk Metrics](#-kpis--risk-metrics)
- [Dashboard & Visualizations](#-dashboard--visualizations)
- [License & Disclaimer](#-license--disclaimer)

---

## 📖 Project Overview

Managing shelter bed capacity and case placement for Unaccompanied Children (UAC) requires continuous coordination between border apprehension velocity, Department of Homeland Security (CBP) holding facilities, and Department of Health and Human Services (HHS) care networks. Delays between custody transfer and sponsor discharge cause severe facility congestion and strain humanitarian operations.

This repository provides an automated data processing, exploratory analysis, and predictive forecasting engine that benchmarks baseline, statistical (ARIMA/SARIMA), and machine learning (Random Forest, Gradient Boosting) architectures under strict chronological validation.

---

## ✨ Key Features

- 🔄 **Automated Preprocessing & Pipeline Normalization:** Parses inverted date formats, strips formatting artifacts, and regularizes calendar reporting gaps using linear interpolation (`asfreq('D')`).
- 📊 **Comprehensive Exploratory Data Analysis (EDA):** Generates 9 publication-grade visualizations, including dual-axis dynamics, correlation matrices, seasonal decompositions (additive, 7-day periodicity), and rolling volatility bands.
- ⚙️ **Leakage-Free Feature Engineering:** Derives multi-period autoregressive lags ($t-1, t-7, t-14$), rolling statistics ($7$ & $14$ days), US federal holiday proxies, and Net Intake Pressure ($\text{Transfers} - \text{Discharges}$).
- 🤖 **Multi-Model Benchmarking:** Out-of-time 80/20 train/test evaluation across 5 distinct algorithms with MAE, RMSE, and MAPE scoring.
- 🚨 **Operational Risk & KPI Engine:** Real-time capacity breach probability calculations ($P(\text{Load} > 7,000)$) via normal CDF modeling and dynamic surge early warning lead times.
- 🖥️ **Interactive Streamlit Web Dashboard:** Stakeholder-ready portal featuring scenario sliders, horizon projections, and metric monitors.

---

## 📂 Project Structure

```text
UAC_Forecasting_Project/
├── data/
│   └── UAC_Data.csv                 # Raw daily time-series records
├── outputs/                         # Generated visual plots & KPI summary tables
│   ├── 01_time_series_dual_axis.png
│   ├── 02_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_monthly_care_load_boxplots.png
│   ├── 05_seasonal_decomposition.png
│   ├── 06_acf_pacf.png
│   ├── 07_transfers_vs_discharges.png
│   ├── 08_net_intake_pressure.png
│   ├── 09_rolling_statistics.png
│   ├── 10_actual_vs_predicted_comparison.png
│   ├── 11_best_model_residuals.png
│   ├── 12_feature_importance.png
│   ├── 13_horizon_error_analysis.png
│   └── kpi_summary.csv
├── src/                             # Modular source components
│   └── __init__.py
├── app.py                           # Interactive Streamlit dashboard application
├── eda_modeling.py                  # Self-contained comprehensive EDA & modeling script
├── executive_summary.md             # Formal report for health operations stakeholders
├── README.md                        # Project documentation & reference manual
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed on your system.
- Standard terminal / PowerShell environment.

### 1. Clone or Navigate to Project Directory
```bash
cd "c:\Users\User\OneDrive\Desktop\css html\UAC_Forecasting_Project"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage Instructions

### Run Complete EDA & Modeling Script
Execute the self-contained pipeline to process data, generate visual charts in `outputs/`, train all 5 models, and print the benchmark results:
```bash
python eda_modeling.py
```

### Launch Interactive Streamlit Dashboard
Launch the monitoring dashboard to interact with scenario forecasts and inspect visual charts:
```bash
streamlit run app.py
```

---

## 📊 Dataset Description

The system ingests daily operational reports tracking unaccompanied minors across custody stages:

| Column Name | Processed Name | Description |
| :--- | :--- | :--- |
| `Date` | `date` | Observation date (`DD-Mon-YY` format). |
| `Children apprehended...*` | `apprehended` | Daily CBP border encounters / apprehensions. |
| `Children in CBP custody` | `in_cbp_custody` | Active census in CBP short-term holding. |
| `Children transferred out...` | `transfers` | Children transferred from CBP to HHS shelter facilities (Inflow). |
| `Children in HHS Care` | `in_hhs_care` | **Target Variable:** Total active census in HHS shelter beds. |
| `Children discharged...` | `discharges` | Minors released from HHS to sponsors / guardians (Outflow). |

---

## 🧠 Model Architecture & Benchmarking

Models are evaluated chronologically on an out-of-time holdout test set (80% train, 20% test):

1. **Moving Average Baseline (7-Day):** Rolling 7-day mean benchmark reflecting naive inertia.
2. **ARIMA $(p, d, q)$:** Statistical autoregressive model fit with automated stationarity parameter selection.
3. **SARIMA $(p, d, q)\times(P, D, Q)_7$:** Seasonal autoregressive model capturing the 7-day administrative cycle.
4. **Random Forest Regressor:** Bagging ensemble (200 estimators) utilizing lag, rolling, and flow features.
5. **Gradient Boosting Regressor:** Sequential boosting ensemble (200 estimators, learning rate = 0.1) yielding state-of-the-art predictive accuracy.

### Evaluated Horizon Errors
- **1-Day Ahead:** MAE $< 20$ children (Tactical bed assignment)
- **7-Day Ahead:** MAE $< 45$ children (Weekly staff & logistics scheduling)
- **14-Day Ahead:** MAE $< 75$ children (Regional shelter reserve activation)

---

## 📈 KPIs & Risk Metrics

- **Forecast Accuracy:** $100 - \text{MAPE}$
- **Net Intake Pressure:** $\text{Transfers} - \text{Discharges}$ (Indicates net accumulation when positive).
- **Capacity Breach Probability:** Modeled as $P(\text{Census} > 7,000) = 1 - \Phi(Z)$ using model residual volatility.
- **Surge Lead Time Buffer:** Estimated time until breach at current 14-day growth velocity.

---

## 📸 Screenshots & Output Gallery

All visualization artifacts are generated automatically into the `outputs/` folder upon running `eda_modeling.py`:

| Output Visualization | Description |
| :--- | :--- |
| `01_time_series_dual_axis.png` | Care load census against daily intake and discharge flows. |
| `03_correlation_heatmap.png` | Cross-correlation between upstream custody and shelter census. |
| `05_seasonal_decomposition.png` | Additive trend, 7-day seasonality, and residual components. |
| `08_net_intake_pressure.png` | Net daily pressure with accumulation and relief bands. |
| `10_actual_vs_predicted_comparison.png` | Test-set comparison of all 5 forecasting models. |
| `12_feature_importance.png` | Gini and relative feature importance for tree ensembles. |

---

## ⚖️ License and Disclaimer

Distributed under the MIT License.

**Disclaimer:** This software and predictive models are developed for educational, analytical, and capacity planning purposes. Healthcare and governmental operations teams should combine model outputs with field intelligence and operational guidelines.
