"""
eda_modeling.py
================
Comprehensive, self-contained Exploratory Data Analysis (EDA) and Predictive Modeling
pipeline for Unaccompanied Children (UAC) Care Load and Placement Demand Forecasting.

Sections:
1. Data Loading & Preprocessing
2. EDA Visualizations
3. Feature Engineering
4. Model Training (Time-based 80/20 split)
5. Model Evaluation & Comparison
6. Risk Analysis & Operational KPIs
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Optional pmdarima / statsmodels fallback for ARIMA/SARIMA
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# Suppress convergence and user warnings for cleaner console output
warnings.filterwarnings("ignore")

# Configure Plotting Style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Create Outputs Directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'UAC_Data.csv')


# ==============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ==============================================================================
print("=" * 80)
print("SECTION 1: DATA LOADING & PREPROCESSING")
print("=" * 80)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at expected path: {DATA_PATH}")

df_raw = pd.read_csv(DATA_PATH)
print(f"[+] Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Rename columns
rename_dict = {
    'Date': 'date',
    'Children apprehended and placed in CBP custody*': 'apprehended',
    'Children in CBP custody': 'in_cbp_custody',
    'Children transferred out of CBP custody': 'transfers',
    'Children in HHS Care': 'in_hhs_care',
    'Children discharged from HHS Care': 'discharges'
}
df_raw = df_raw.rename(columns=rename_dict)

# Clean numeric values (remove commas and quotes, parse ints)
numeric_cols = ['apprehended', 'in_cbp_custody', 'transfers', 'in_hhs_care', 'discharges']
for col in numeric_cols:
    if col in df_raw.columns:
        if df_raw[col].dtype == object:
            df_raw[col] = df_raw[col].astype(str).str.replace(',', '').str.strip()
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Parse date with dayfirst=True and sort ascending
df_raw['date'] = pd.to_datetime(df_raw['date'], format='%d-%b-%y', errors='coerce')
if df_raw['date'].isna().any():
    # Fallback to general parsing if any format mismatch occurs
    df_raw['date'] = pd.to_datetime(df_raw['date'], dayfirst=True)

df_clean = df_raw.sort_values('date', ascending=True).reset_index(drop=True)
df_clean = df_clean.set_index('date')

print(f"[+] Date range in raw data: {df_clean.index.min().strftime('%Y-%m-%d')} to {df_clean.index.max().strftime('%Y-%m-%d')}")
print(f"[+] Raw data records: {len(df_clean)}")

# Regularize daily frequency & linearly interpolate missing days
df_daily = df_clean.asfreq('D')
missing_days_count = df_daily.isna().sum()['in_hhs_care']
print(f"[+] Missing calendar days interpolated via asfreq('D'): {missing_days_count} days")

df_interpolated = df_daily.interpolate(method='linear')
# Fill any leading edge if interpolation left any nulls
df_interpolated = df_interpolated.bfill().ffill()

# Cast values to integer where appropriate for daily counts
df_interpolated = df_interpolated.round().astype(int)

print("\n--- Processed Dataset Summary ---")
print(f"Shape: {df_interpolated.shape}")
print("\nData Types:")
print(df_interpolated.dtypes)
print("\nFirst 5 Records:")
print(df_interpolated.head())
print("\nDescriptive Statistics:")
print(df_interpolated.describe())
print("\nMissing Values Count:")
print(df_interpolated.isna().sum())


# ==============================================================================
# SECTION 2: EDA VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 2: EDA VISUALIZATIONS (Saving figures to outputs/)")
print("=" * 80)

# 1. Dual y-axis time series plot
fig, ax1 = plt.subplots(figsize=(14, 6))
color_care = '#1f77b4'
ax1.set_title('UAC Care Load vs Flow Dynamics Over Time', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Children in HHS Care (Census)', color=color_care, fontsize=12, fontweight='bold')
ax1.plot(df_interpolated.index, df_interpolated['in_hhs_care'], color=color_care, linewidth=2, label='In HHS Care (Total Load)')
ax1.tick_params(axis='y', labelcolor=color_care)

ax2 = ax1.twinx()
ax2.set_ylabel('Daily Inflow / Outflow Counts', color='#333333', fontsize=12, fontweight='bold')
ax2.plot(df_interpolated.index, df_interpolated['transfers'], color='#2ca02c', alpha=0.6, label='Transfers into HHS (Inflow)')
ax2.plot(df_interpolated.index, df_interpolated['discharges'], color='#d62728', alpha=0.6, label='Discharges from HHS (Outflow)')
ax2.plot(df_interpolated.index, df_interpolated['apprehended'], color='#ff7f0e', alpha=0.4, linestyle='--', label='CBP Apprehensions')
ax2.tick_params(axis='y', labelcolor='#333333')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01_time_series_dual_axis.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 01_time_series_dual_axis.png")

# 2. Distribution plots (Histograms + KDE)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

for idx, col in enumerate(numeric_cols):
    sns.histplot(df_interpolated[col], kde=True, ax=axes[idx], color=palette[idx % len(palette)], bins=30)
    axes[idx].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Density / Frequency')

axes[5].axis('off')
fig.suptitle('Empirical Distributions & Density Functions of UAC Variables', fontsize=15, fontweight='bold', y=0.98)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_distributions.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 02_distributions.png")

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df_interpolated[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Cross-Correlation Matrix of UAC Pipeline Variables', fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 03_correlation_heatmap.png")

# 4. Box plots by month for care load
fig, ax = plt.subplots(figsize=(12, 6))
df_monthly = df_interpolated.copy()
df_monthly['month_name'] = df_monthly.index.strftime('%b')
df_monthly['month_num'] = df_monthly.index.month
df_monthly = df_monthly.sort_values('month_num')

sns.boxplot(x='month_name', y='in_hhs_care', data=df_monthly, palette='crest', ax=ax)
ax.set_title('Monthly Distribution & Seasonality of HHS Care Load', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Children in HHS Care', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04_monthly_care_load_boxplots.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 04_monthly_care_load_boxplots.png")

# 5. Seasonal Decomposition (additive, period=7)
decomp = seasonal_decompose(df_interpolated['in_hhs_care'], model='additive', period=7)
fig, (ax_obs, ax_trend, ax_seas, ax_resid) = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
ax_obs.plot(decomp.observed, color='#1f77b4', lw=1.5)
ax_obs.set_ylabel('Observed', fontweight='bold')
ax_obs.set_title('Additive Time Series Decomposition (Period = 7 Days)', fontsize=14, fontweight='bold')

ax_trend.plot(decomp.trend, color='#ff7f0e', lw=1.5)
ax_trend.set_ylabel('Trend', fontweight='bold')

ax_seas.plot(decomp.seasonal, color='#2ca02c', lw=1.2)
ax_seas.set_ylabel('Seasonal (7d)', fontweight='bold')

ax_resid.scatter(decomp.resid.index, decomp.resid, color='#d62728', s=10, alpha=0.7)
ax_resid.axhline(0, color='black', linestyle='--', alpha=0.6)
ax_resid.set_ylabel('Residual', fontweight='bold')
ax_resid.set_xlabel('Date', fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05_seasonal_decomposition.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 05_seasonal_decomposition.png")

# 6. ACF and PACF plots
fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df_interpolated['in_hhs_care'], lags=30, ax=ax_acf, title='Autocorrelation Function (ACF) - HHS Care')
plot_pacf(df_interpolated['in_hhs_care'], lags=30, ax=ax_pacf, title='Partial Autocorrelation (PACF) - HHS Care', method='yule_walker')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_acf_pacf.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 06_acf_pacf.png")

# 7. Transfers vs Discharges Scatter Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x='transfers', y='discharges', data=df_interpolated,
            scatter_kws={'alpha': 0.5, 'color': '#1f77b4', 's': 30},
            line_kws={'color': '#d62728', 'linewidth': 2}, ax=ax)
max_val = max(df_interpolated['transfers'].max(), df_interpolated['discharges'].max())
ax.plot([0, max_val], [0, max_val], color='gray', linestyle=':', label='Breakeven Inflow = Outflow')
ax.set_title('Transfers into HHS vs Discharges from HHS', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Transfers from CBP into HHS (Daily Inflow)', fontsize=11)
ax.set_ylabel('Discharges from HHS Care (Daily Outflow)', fontsize=11)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07_transfers_vs_discharges.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 07_transfers_vs_discharges.png")

# 8. Net intake pressure over time
df_interpolated['net_intake_pressure'] = df_interpolated['transfers'] - df_interpolated['discharges']
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_interpolated.index, df_interpolated['net_intake_pressure'], color='#333333', lw=1, alpha=0.5, label='Daily Net Pressure')
ax.plot(df_interpolated.index, df_interpolated['net_intake_pressure'].rolling(14).mean(), color='#e65100', lw=2.2, label='14-Day Rolling Net Pressure')
ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Net Balance Threshold (Zero)')
ax.fill_between(df_interpolated.index, df_interpolated['net_intake_pressure'], 0,
                where=(df_interpolated['net_intake_pressure'] > 0), color='#ff9999', alpha=0.3, label='Accumulation Period')
ax.fill_between(df_interpolated.index, df_interpolated['net_intake_pressure'], 0,
                where=(df_interpolated['net_intake_pressure'] < 0), color='#99ff99', alpha=0.3, label='Relief / Decongestion Period')
ax.set_title('Net Operational Intake Pressure (Transfers minus Discharges)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Net Daily Change (Children)', fontsize=11)
ax.legend(loc='upper right', frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '08_net_intake_pressure.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 08_net_intake_pressure.png")

# 9. Rolling Statistics with Confidence Bands
rolling_7 = df_interpolated['in_hhs_care'].rolling(7)
rolling_14 = df_interpolated['in_hhs_care'].rolling(14)
mean_7 = rolling_7.mean()
std_7 = rolling_7.std()
mean_14 = rolling_14.mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_interpolated.index, df_interpolated['in_hhs_care'], color='#888888', alpha=0.4, label='Actual Care Load', lw=1.2)
ax.plot(df_interpolated.index, mean_7, color='#1f77b4', lw=2, label='7-Day Rolling Mean')
ax.fill_between(df_interpolated.index, mean_7 - 1.96 * std_7, mean_7 + 1.96 * std_7, color='#1f77b4', alpha=0.15, label='7-Day ± 1.96σ Band')
ax.plot(df_interpolated.index, mean_14, color='#e65100', lw=2, linestyle='--', label='14-Day Rolling Mean')
ax.set_title('HHS Care Load Rolling Trajectories & Volatility Bands', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Children in HHS Care', fontsize=11)
ax.legend(loc='upper left', frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '09_rolling_statistics.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 09_rolling_statistics.png")


# ==============================================================================
# SECTION 3: FEATURE ENGINEERING
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 3: FEATURE ENGINEERING")
print("=" * 80)

df_feat = df_interpolated.copy()

# 1. Lag features of target
df_feat['lag_1'] = df_feat['in_hhs_care'].shift(1)
df_feat['lag_7'] = df_feat['in_hhs_care'].shift(7)
df_feat['lag_14'] = df_feat['in_hhs_care'].shift(14)

# 2. Rolling features (using shift(1) to strictly prevent data leakage from target)
df_feat['rolling_mean_7'] = df_feat['in_hhs_care'].shift(1).rolling(7).mean()
df_feat['rolling_mean_14'] = df_feat['in_hhs_care'].shift(1).rolling(14).mean()
df_feat['rolling_var_7'] = df_feat['in_hhs_care'].shift(1).rolling(7).var()
df_feat['rolling_var_14'] = df_feat['in_hhs_care'].shift(1).rolling(14).var()

# 3. Flow feature
# net_intake_pressure already calculated; shift by 1 to reflect known past pressure
df_feat['net_intake_pressure'] = (df_feat['transfers'] - df_feat['discharges']).shift(1)

# 4. Calendar features
df_feat['day_of_week'] = df_feat.index.dayofweek
df_feat['month'] = df_feat.index.month
df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)

# Holiday proxy: using holidays package or federal proxy
if HAS_HOLIDAYS:
    us_holidays = holidays.US(years=range(df_feat.index.year.min(), df_feat.index.year.max() + 1))
    df_feat['is_holiday_proxy'] = df_feat.index.map(lambda d: 1 if d in us_holidays else 0)
else:
    # Proxy: New Year, MLK, Presidents Day, Memorial Day, Juneteenth, July 4, Labor Day, Veterans, Thanksgiving, Christmas
    df_feat['is_holiday_proxy'] = ((df_feat['month'] == 1) & (df_feat.index.day == 1) |
                                   (df_feat['month'] == 7) & (df_feat.index.day == 4) |
                                   (df_feat['month'] == 12) & (df_feat.index.day == 25)).astype(int)

# Drop initial rows with NaN from rolling and lag operations
df_model_ready = df_feat.dropna().copy()

feature_columns = [
    'lag_1', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_14',
    'rolling_var_7', 'rolling_var_14',
    'net_intake_pressure',
    'day_of_week', 'month', 'is_weekend', 'is_holiday_proxy'
]

print(f"[+] Total samples ready for modeling after lag truncation: {len(df_model_ready)}")
print("\n--- Feature Set Summary ---")
feat_summary = pd.DataFrame({
    'Feature': feature_columns,
    'Type': [str(df_model_ready[col].dtype) for col in feature_columns],
    'Mean': [df_model_ready[col].mean() for col in feature_columns],
    'Std': [df_model_ready[col].std() for col in feature_columns],
    'Min': [df_model_ready[col].min() for col in feature_columns],
    'Max': [df_model_ready[col].max() for col in feature_columns]
})
print(feat_summary.to_string(index=False))


# ==============================================================================
# SECTION 4: MODEL TRAINING (TIME-BASED 80/20 SPLIT)
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 4: MODEL TRAINING (Chronological 80/20 Split)")
print("=" * 80)

split_idx = int(len(df_model_ready) * 0.8)
train_df = df_model_ready.iloc[:split_idx]
test_df = df_model_ready.iloc[split_idx:]

print(f"[+] Training Set: {len(train_df)} observations ({train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')})")
print(f"[+] Testing Set:  {len(test_df)} observations ({test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})")

X_train = train_df[feature_columns]
y_train = train_df['in_hhs_care']
X_test = test_df[feature_columns]
y_test = test_df['in_hhs_care']

predictions = {}

# 1. Baseline Model: 7-day Moving Average of target
print("\n[1/5] Training 7-Day Moving Average Baseline...")
# Predicted value for t is the mean of the past 7 days (i.e. rolling_mean_7)
predictions['Moving Average (7d)'] = test_df['rolling_mean_7'].values

# 2. Statistical Model: ARIMA
print("[2/5] Training ARIMA Model...")
if HAS_PMDARIMA:
    arima_fit = pm.auto_arima(
        train_df['in_hhs_care'],
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=1,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    arima_preds = arima_fit.predict(n_periods=len(test_df))
    print(f"      Selected ARIMA Order: {arima_fit.order}")
else:
    arima_model = ARIMA(train_df['in_hhs_care'], order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_preds = arima_fit.forecast(steps=len(test_df))
predictions['ARIMA'] = np.array(arima_preds)

# 3. Statistical Model: SARIMA (m=7 for weekly seasonality)
print("[3/5] Training SARIMA Model (m=7)...")
if HAS_PMDARIMA:
    try:
        sarima_fit = pm.auto_arima(
            train_df['in_hhs_care'],
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            d=1,
            seasonal=True,
            m=7,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        sarima_preds = sarima_fit.predict(n_periods=len(test_df))
        print(f"      Selected SARIMA Order: {sarima_fit.order}x{sarima_fit.seasonal_order}")
    except Exception as e:
        print(f"      SARIMA auto fit failed: {e}. Falling back to default SARIMAX(1,1,1)x(1,1,1,7).")
        sarima_model = SARIMAX(train_df['in_hhs_care'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_preds = sarima_fit.forecast(steps=len(test_df))
else:
    sarima_model = SARIMAX(train_df['in_hhs_care'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_preds = sarima_fit.forecast(steps=len(test_df))
predictions['SARIMA'] = np.array(sarima_preds)

# 4. Machine Learning Model: Random Forest Regressor
print("[4/5] Training Random Forest Regressor (n_estimators=200)...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
predictions['Random Forest'] = rf_model.predict(X_test)

# 5. Machine Learning Model: Gradient Boosting Regressor
print("[5/5] Training Gradient Boosting Regressor (n_estimators=200, lr=0.1)...")
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
predictions['Gradient Boosting'] = gb_model.predict(X_test)

print("[✓] Model Training Completed for all 5 architectures.")


# ==============================================================================
# SECTION 5: MODEL EVALUATION & COMPARISON
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 5: MODEL EVALUATION & COMPARISON")
print("=" * 80)

metrics_list = []
y_true = y_test.values

for name, y_pred in predictions.items():
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    acc = 100.0 - mape
    metrics_list.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'Accuracy (%)': acc
    })

results_df = pd.DataFrame(metrics_list).sort_values('MAE').reset_index(drop=True)
print("\n--- Model Performance Comparison (Sorted by MAE) ---")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

best_model_name = results_df.iloc[0]['Model']
best_predictions = predictions[best_model_name]
print(f"\n[★] Top Performing Architecture: {best_model_name} (MAE: {results_df.iloc[0]['MAE']:.2f}, MAPE: {results_df.iloc[0]['MAPE (%)']:.2f}%)")

# 1. Actual vs Predicted Plot for each model
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(test_df.index, y_true, color='black', linewidth=2.5, label='Actual HHS Care Load', zorder=5)
palette_models = {'Moving Average (7d)': '#999999', 'ARIMA': '#ff7f0e', 'SARIMA': '#2ca02c', 'Random Forest': '#1f77b4', 'Gradient Boosting': '#d62728'}

for name, y_pred in predictions.items():
    lw = 2.0 if name == best_model_name else 1.3
    ls = '-' if 'Forest' in name or 'Boosting' in name else '--'
    ax.plot(test_df.index, y_pred, label=f"{name} (MAE: {results_df[results_df['Model']==name]['MAE'].values[0]:.1f})",
            linewidth=lw, linestyle=ls, color=palette_models.get(name, None))

ax.set_title('Test Set Forecasting Evaluation: Actual vs Model Predictions', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Children in HHS Care', fontsize=11)
ax.legend(loc='upper right', frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '10_actual_vs_predicted_comparison.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 10_actual_vs_predicted_comparison.png")

# 2. Residuals Plot for Best Model
best_residuals = y_true - best_predictions
fig, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(14, 5))

ax_res1.plot(test_df.index, best_residuals, color='#1f77b4', lw=1.5)
ax_res1.axhline(0, color='red', linestyle='--', lw=1.5)
ax_res1.set_title(f'Residuals Over Time ({best_model_name})', fontsize=12, fontweight='bold')
ax_res1.set_xlabel('Date')
ax_res1.set_ylabel('Residual Error (Actual - Predicted)')

sns.histplot(best_residuals, kde=True, color='#1f77b4', ax=ax_res2)
ax_res2.axvline(0, color='red', linestyle='--', lw=1.5)
ax_res2.set_title(f'Residual Distribution & Normality ({best_model_name})', fontsize=12, fontweight='bold')
ax_res2.set_xlabel('Residual Error')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '11_best_model_residuals.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 11_best_model_residuals.png")

# 3. Feature Importance Chart (Random Forest & Gradient Boosting)
fig, (ax_rf, ax_gb) = plt.subplots(1, 2, figsize=(15, 6))

rf_feat_imp = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=True)
rf_feat_imp.plot(kind='barh', color='#1f77b4', ax=ax_rf)
ax_rf.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
ax_rf.set_xlabel('Relative Importance (Gini)')

gb_feat_imp = pd.Series(gb_model.feature_importances_, index=feature_columns).sort_values(ascending=True)
gb_feat_imp.plot(kind='barh', color='#d62728', ax=ax_gb)
ax_gb.set_title('Gradient Boosting Feature Importance', fontsize=12, fontweight='bold')
ax_gb.set_xlabel('Relative Importance')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '12_feature_importance.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 12_feature_importance.png")

# 4. Multi-Horizon Error Analysis (1-day, 7-day, 14-day error progression)
horizons = [1, 7, 14]
horizon_errors = []

for h in horizons:
    if len(y_true) >= h:
        mae_h = mean_absolute_error(y_true[:h], best_predictions[:h])
        rmse_h = np.sqrt(mean_squared_error(y_true[:h], best_predictions[:h]))
        mape_h = mean_absolute_percentage_error(y_true[:h], best_predictions[:h]) * 100
        horizon_errors.append({'Horizon (Days)': f"{h}-Day Ahead", 'MAE': mae_h, 'RMSE': rmse_h, 'MAPE (%)': mape_h})

horizon_df = pd.DataFrame(horizon_errors)
print("\n--- Forecast Horizon Error Analysis (Best Model) ---")
print(horizon_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

fig, ax_h = plt.subplots(figsize=(7, 4.5))
bars = ax_h.bar(horizon_df['Horizon (Days)'], horizon_df['MAE'], color=['#42a5f5', '#1e88e5', '#0d47a1'], width=0.5)
ax_h.set_title(f'Mean Absolute Error across Forecast Horizons ({best_model_name})', fontsize=12, fontweight='bold', pad=10)
ax_h.set_ylabel('MAE (Children in Care)')
for bar in bars:
    yval = bar.get_height()
    ax_h.text(bar.get_x() + bar.get_width()/2.0, yval + (0.02 * yval), f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '13_horizon_error_analysis.png'), dpi=300)
plt.close(fig)
print("[✓] Saved 13_horizon_error_analysis.png")


# ==============================================================================
# SECTION 6: RISK ANALYSIS & OPERATIONAL KPIS
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 6: RISK ANALYSIS & OPERATIONAL KPIS")
print("=" * 80)

# 1. Operational Parameters & Thresholds
CAPACITY_THRESHOLD = 7000  # Statutory / Operational surge threshold (Children)
latest_actual = df_interpolated['in_hhs_care'].iloc[-1]
latest_pred = best_predictions[-1]

best_mape = results_df.loc[results_df['Model'] == best_model_name, 'MAPE (%)'].values[0]
forecast_accuracy = 100.0 - best_mape

# Capacity Breach Probability using normal CDF based on residual standard deviation
residual_std = np.std(best_residuals)
if residual_std == 0:
    residual_std = 1.0

# Probability P(Care Load > 7000 | Forecast)
# Z = (Capacity - Forecast) / sigma
z_score = (CAPACITY_THRESHOLD - latest_pred) / residual_std
capacity_breach_prob = (1.0 - stats.norm.cdf(z_score)) * 100.0

# Surge Lead Time Estimation
# Calculate recent 14-day net growth rate (velocity = dLoad/dt)
recent_14_growth = (df_interpolated['in_hhs_care'].iloc[-1] - df_interpolated['in_hhs_care'].iloc[-15]) / 14.0
if recent_14_growth > 0:
    days_to_breach = (CAPACITY_THRESHOLD - latest_actual) / recent_14_growth
    surge_lead_time_desc = f"{days_to_breach:.1f} days at current trajectory (+{recent_14_growth:.1f}/day)"
else:
    surge_lead_time_desc = f"Stable / Negative Net Growth ({recent_14_growth:.1f}/day; >90 days cushion)"

print(f"[+] Key Performance Indicators (KPIs):")
print(f"    - Current HHS Care Load:            {latest_actual:,} children")
print(f"    - Current Forecasted Load:          {int(round(latest_pred)):,} children")
print(f"    - Best Model Forecast Accuracy:     {forecast_accuracy:.2f}% (MAPE: {best_mape:.2f}%)")
print(f"    - Operational Capacity Threshold:   {CAPACITY_THRESHOLD:,} children")
print(f"    - Capacity Breach Probability:      {capacity_breach_prob:.4f}%")
print(f"    - Surge Early Warning Lead Time:    {surge_lead_time_desc}")

# Save KPI Summary File for Stakeholders
kpi_summary_df = pd.DataFrame([
    {'Metric': 'Current Active HHS Care Load', 'Value': f"{latest_actual:,} children"},
    {'Metric': 'Next Period Forecast (Best Model)', 'Value': f"{int(round(latest_pred)):,} children"},
    {'Metric': 'Optimal Architecture', 'Value': best_model_name},
    {'Metric': 'Best Model Accuracy', 'Value': f"{forecast_accuracy:.2f}%"},
    {'Metric': 'Best Model MAE', 'Value': f"{results_df.iloc[0]['MAE']:.2f}"},
    {'Metric': 'Operational Surge Threshold', 'Value': f"{CAPACITY_THRESHOLD:,} beds"},
    {'Metric': 'Capacity Breach Risk P(Load > 7000)', 'Value': f"{capacity_breach_prob:.4f}%"},
    {'Metric': 'Surge Lead Time Buffer', 'Value': surge_lead_time_desc}
])

kpi_csv_path = os.path.join(OUTPUT_DIR, 'kpi_summary.csv')
kpi_summary_df.to_csv(kpi_csv_path, index=False)
print(f"[✓] Saved KPI summary table to: {kpi_csv_path}")

print("\n" + "=" * 80)
print("EXECUTION COMPLETED SUCCESSFULLY. ALL ARTIFACTS GENERATED.")
print("=" * 80)
