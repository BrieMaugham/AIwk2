# sdg13_co2_regression.py
# Mini Project: SDG 13 - Forecast CO2 emissions (supervised regression)
# Requirements: pandas numpy scikit-learn matplotlib seaborn joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------
# 1. Download dataset
# -----------------------
DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
print("Downloading dataset...")
df = pd.read_csv(DATA_URL, low_memory=False)
print("Rows:", df.shape[0], "Cols:", df.shape[1])

# -----------------------
# 2. Quick exploration
# -----------------------
print("\nColumns preview:", df.columns.tolist()[:20])
print(df[['iso_code','country','year','co2','population','gdp']].head())

# We will focus on country-level data (exclude continents and aggregates)
# Filter: iso_code not null and iso_code length == 3 (country codes)
df = df[df['iso_code'].apply(lambda x: isinstance(x, str) and len(x) == 3)].copy()

# Select a subset of columns likely to be predictive (and present for many countries)
cols = ['iso_code','country','year','co2','population','gdp','gdp_per_capita',
        'primary_energy_consumption','energy_per_capita','cement_co2','coal_co2','oil_co2','gas_co2']
# Some columns might not exist in older versions - filter existing
cols = [c for c in cols if c in df.columns]
df = df[cols].copy()
print("\nUsing columns:", cols)

# -----------------------
# 3. Preprocessing
# -----------------------
# Keep recent years to reduce missingness; for the prototype use 1990-2019 inclusive
df = df[(df['year'] >= 1990) & (df['year'] <= 2019)].copy()

# Fill missing numeric values with country median for each feature (simple approach)
numeric_cols = [c for c in df.columns if c not in ['iso_code','country','year','co2']]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Country-level imputation: fill na with country median, then global median
for c in numeric_cols:
    df[c] = df.groupby('iso_code')[c].transform(lambda x: x.fillna(x.median()))
    df[c] = df[c].fillna(df[c].median())

# Drop rows without target (co2)
df['co2'] = pd.to_numeric(df['co2'], errors='coerce')
df = df.dropna(subset=['co2']).copy()

# Feature: log transforms often help with skewed distributions
for f in ['population','gdp','gdp_per_capita','primary_energy_consumption','energy_per_capita','cement_co2','coal_co2','oil_co2','gas_co2']:
    if f in df.columns:
        df[f + '_log'] = np.log1p(df[f])

# Add year as a feature (numeric)
df['year_norm'] = df['year'] - df['year'].min()

# Choose features for model
feature_cols = [c for c in df.columns if c.endswith('_log') or c == 'year_norm']
print("Feature columns:", feature_cols)

# Target
y = df['co2'].values
X = df[feature_cols].values

# Simple train-test split (stratify by year? we'll random split but keep country variety)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# -----------------------
# 4. Baseline model: Linear Regression
# -----------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

def evaluate(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{label} -- MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
    return {'mae':mae,'rmse':rmse,'r2':r2}

eval_lr = evaluate(y_test, y_pred_lr, "LinearRegression")

# -----------------------
# 5. Random Forest Regressor (stronger model)
# -----------------------
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
eval_rf = evaluate(y_test, y_pred_rf, "RandomForest")

# Feature importance for interpretability
importances = rf.feature_importances_
fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
print("\nTop features by importance:")
for f, imp in fi[:10]:
    print(f, imp)

# -----------------------
# 6. Visualizations
# -----------------------
# Residual plot for RF
resid = y_test - y_pred_rf
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred_rf, y=resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted CO2')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (RandomForest)')
plt.tight_layout()
plt.savefig('residuals_rf.png', dpi=150)

# True vs Predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True CO2')
plt.ylabel('Predicted CO2')
plt.title('True vs Predicted (RandomForest)')
plt.tight_layout()
plt.savefig('true_vs_pred_rf.png', dpi=150)

# -----------------------
# 7. Save model and preprocessing metadata
# -----------------------
os.makedirs('model', exist_ok=True)
joblib.dump(rf, 'model/rf_co2_model.joblib')
joblib.dump(lr, 'model/lr_co2_model.joblib')

# Save selected feature list
import json
with open('model/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

print("\nModels saved to ./model/. Visualizations exported.")
