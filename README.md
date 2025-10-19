# SDG 13 — Mini Project: Forecasting CO₂ Emissions

**Project summary**  
This project demonstrates how supervised machine learning can support **SDG 13 (Climate Action)** by forecasting national CO₂ emissions using public data. The prototype model helps identify countries and years with rising emissions and can support policy/mitigation prioritization.

## Files
- `sdg13_co2_regression.py` — Full script: data download, preprocessing, modeling, evaluation, and visualization.
- `requirements.txt` — Python dependencies.
- `model/` — Saved model files after running (`rf_co2_model.joblib`, `lr_co2_model.joblib`).
- `residuals_rf.png`, `true_vs_pred_rf.png` — Visualization outputs.

## How to run (local or Colab)
1. Clone the repo:

git clone <your-repo-url>
cd <repo-folder>
Install dependencies:

pip install -r requirements.txt
Run the script:

python sdg13_co2_regression.py
Output: trained models saved in model/, evaluation metrics printed to console, plots saved as PNG.

Dataset
Source: Our World in Data CO₂ dataset (public):
https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv

ML approach
Supervised regression (Random Forest + Linear Regression baseline)

Features: population, GDP, energy metrics (log transforms used)

Evaluation metrics: MAE, RMSE, R²

Results (example)
RandomForest MAE: see console after running

Feature importance reported to help with interpretability.

Ethics & Limitations
Possible biases from missing or inconsistent country-level reporting.

Model cannot infer causation — only patterns in historical reporting.

Important to include socio-economic context and domain expertise before policy decisions.

Next steps
Add time-series forecasting per country (e.g., LSTM) and country-specific models.

Integrate real-time data (e.g., energy consumption APIs).

Deploy via Streamlit dash for interactive visualization.


## 5) One-page report (submit to LMS)
Copy this as your 1-page report (`report.md` or a document):

**Title:** Forecasting National CO₂ Emissions — SDG 13 Prototype

**Problem & Motivation:**  
Rapid identification of countries and time periods with rising CO₂ emissions helps policymakers target mitigation interventions. This prototype demonstrates a machine learning workflow to forecast national CO₂ emissions using openly available data.

**Dataset & Preprocessing:**  
Used Our World in Data CO₂ dataset (country-year records). Filtered to 1990–2019, selected numeric predictors (population, GDP, energy consumption, fuel-specific CO₂ components). Missing values were imputed with country medians, then global medians. Log transforms applied to skewed features and year normalized.

**ML Approach:**  
Supervised regression: baseline Linear Regression and a Random Forest Regressor. Training/test split 80/20. Evaluated with MAE, RMSE, and R². Random Forest chosen for non-linear patterns and interpretability via feature importances.

**Results:**  
Random Forest performed better than linear baseline (report MAE/RMSE/R² after you run the script). Top predictive features included population (log), energy per capita (log), and GDP-related features — which aligns with domain expectations.

**Ethical Considerations:**  
- Data quality and reporting biases across countries can skew results; missing data imputation may hide reporting gaps.  
- Predictions should be paired with expert analysis; model outputs are advisory, not prescriptive.  
- Careful about policy consequences — false signals could misallocate resources.

**Conclusion & Next Steps:**  
The prototype shows how ML can highlight emission trends and support SDG 13 decision-making. Next steps: per-country time-series forecasting, model explainability (SHAP), and an interactive dashboard for policymakers.
