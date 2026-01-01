# Hospital Readmission Prediction Analysis

A comprehensive machine learning analysis for predicting 30-day hospital readmissions using patient demographic, clinical, and operational data.

## Overview

This project analyzes hospital readmission patterns across 10,000 patient records to identify key risk factors and build predictive models. The analysis focuses on understanding which patients are at highest risk of readmission and what factors contribute most significantly to readmission risk.

## Key Findings

- **Overall readmission rate**: 15.47%
- **Highest-risk diagnosis**: Heart Failure (22.3% readmission rate)
- **Age impact**: Patients over 50 have a 20.37% readmission rate
- **Top predictors**: Age, follow-up scheduling, ICU admission, and length of stay

## Dataset

The analysis uses `hospital - hospital.csv` containing 18 features:

- **Demographics**: Age, Gender, Insurance Type, Marital Status
- **Clinical**: Primary Diagnosis, BMI, Glucose Level, Comorbidities
- **Operational**: Department, Admission Type, ICU Admission, Surgery, Length of Stay (LOS)
- **Historical**: Previous Admissions (12M), ER Visits
- **Post-discharge**: Follow-up Scheduled, Home Health Referral
- **Target**: Readmitted (binary outcome)

## Methodology

### 1. Exploratory Data Analysis
- Readmission patterns by department, diagnosis, age group, and length of stay
- Correlation analysis between follow-up scheduling and readmission rates
- Missing value assessment and data quality checks

### 2. Predictive Modeling
- **Logistic Regression** (baseline): AUROC 0.61, AUPRC 0.38
- **Random Forest**: Enhanced performance with balanced class weights
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **Ensemble Model**: Combined predictions for improved accuracy

### 3. Feature Importance Analysis
- SHAP value computation for model interpretability
- Coefficient-based importance for logistic regression
- Tree-based importance for ensemble models

### 4. Survival Analysis
- Kaplan-Meier curves for time-to-readmission
- Department-level survival comparisons
- Simulated time-to-event analysis

## Repository Structure

```
├── REN_Analytics_Comp.ipynb    # Main analysis notebook
├── MSBA-Presentation.pdf       # Project presentation
└── README.md                   # This file
```

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

## Usage

1. Ensure the hospital dataset is available: `hospital - hospital.csv`[https://github.com/codymbaldwin/sample-files/blob/master/hospital_ratings.csv]
2. Open `REN_Analytics_Comp.ipynb` in Jupyter Notebook or Google Colab
3. Run cells sequentially to reproduce the analysis
4. Generated visualizations include:
   - Readmission rates by department, diagnosis, age, and LOS
   - ROC curves and confusion matrices
   - Feature importance plots
   - Kaplan-Meier survival curves
   - SHAP summary plots

## Key Insights

1. **Follow-up scheduling** has a strong inverse relationship with readmission (scheduled follow-ups reduce readmission risk)
2. **Heart failure and COPD** patients show significantly higher readmission rates
3. **Age** is the strongest predictor, with older patients at higher risk
4. **General Medicine and Cardiology** departments have the highest readmission volumes
5. **Length of stay** shows a non-linear relationship with readmission risk

## Model Performance

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Logistic Regression | 0.6111 | 0.3750 |
| Random Forest | Improved | Improved |
| XGBoost | Best single model | Best single model |
| Ensemble | Optimal | Optimal |

## Contributors

MSBA Analytics Competition Team

## License

This project is for educational and analytical purposes.
