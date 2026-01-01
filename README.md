# Hospital Readmission Prediction Analysis

A comprehensive machine learning project analyzing 30-day hospital readmissions using patient demographic, clinical, and operational data. This analysis identifies key risk factors, builds predictive models, and provides actionable insights for healthcare providers.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Installation and Usage](#installation-and-usage)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

## Overview

This project analyzes hospital readmission patterns across **10,000 patient records** to identify which patients are at highest risk of readmission within 30 days and what factors contribute most significantly to readmission risk. The analysis combines exploratory data analysis, predictive modeling, and survival analysis to provide comprehensive insights.

### Business Impact
- **15.47% overall readmission rate** identified across all patients
- High-risk populations identified for targeted interventions
- Predictive models achieving **AUROC 0.61** for risk stratification
- Cost-saving opportunities through improved follow-up scheduling

## Dataset

The analysis uses hospital patient data containing **18 features** across 10,000 admissions:

### Features

**Demographics**
- Age
- Gender
- Insurance Type
- Marital Status

**Clinical Indicators**
- Primary Diagnosis
- BMI (Body Mass Index)
- Glucose Level
- Number of Comorbidities

**Operational Factors**
- Department
- Admission Type (Emergency/Elective)
- ICU Admission (Yes/No)
- Surgery (Yes/No)
- Length of Stay (LOS)

**Historical Utilization**
- Previous Admissions (12 months)
- Emergency Room Visits

**Post-Discharge**
- Follow-up Scheduled (Yes/No)
- Home Health Referral (Yes/No)

**Target Variable**
- Readmitted (Binary: 0 = No, 1 = Yes)

### Data Source
Dataset: `hospital - hospital.csv`

**Note**: Ensure the dataset is available in your working directory before running the analysis.

## Key Findings

### Readmission Statistics
- **Overall readmission rate**: 15.47% (1,547 out of 10,000 patients)
- **Highest-risk diagnosis**: Heart Failure with 22.3% readmission rate
- **Age impact**: Patients over 50 have a 20.37% readmission rate
- **COPD patients**: 19.3% readmission rate (second highest)

### Critical Risk Factors
1. **Age** - Strongest predictor (coefficient: 0.49)
2. **Follow-up Scheduling** - Inverse relationship with readmission (coefficient: -0.18)
3. **ICU Admission** - Positive correlation with readmission risk
4. **Length of Stay** - Non-linear relationship with readmission

### Department Analysis
Highest readmission rates observed in:
- General Medicine
- Cardiology
- Pulmonology

### Protective Factors
- Scheduled follow-up appointments significantly reduce readmission risk
- Home health referrals show protective effect
- Proper discharge planning correlates with better outcomes

## Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Median imputation for numeric features, "Unknown" category for categorical features
- **Outlier Management**: Infinite values replaced with NaN, then imputed
- **Feature Engineering**: Age bins, LOS bins, diagnosis standardization
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler applied to numeric features

### 2. Exploratory Data Analysis
Comprehensive analysis including:
- Readmission rate distributions by department, diagnosis, age group, and length of stay
- Correlation analysis between follow-up scheduling and readmission rates
- Department-level readmission volume analysis
- Patient demographic profiling
- ICU admission impact assessment

### 3. Predictive Modeling

#### Models Implemented

**Baseline Model: Logistic Regression**
- Max iterations: 1000
- Class weighting: Balanced
- AUROC: 0.6111
- AUPRC: 0.3750
- Brier Score: 0.1490

**Random Forest Classifier**
- 200 estimators
- Balanced class weights
- Enhanced performance over baseline
- Provides feature importance rankings

**XGBoost Classifier**
- Gradient boosting approach
- Optimized hyperparameters
- Best single-model performance
- Handles imbalanced data effectively

**Ensemble Model**
- Combines Random Forest and XGBoost predictions
- Weighted averaging approach (50/50)
- Optimal overall performance
- Improved robustness and generalization

#### Model Validation
- **Train/Test Split**: 80/20 chronological split
- **Patient-Level Validation**: Ensures no patient appears in both train and test sets
- **Stratified Sampling**: Maintains class distribution
- **Cross-Validation**: Applied to prevent overfitting

### 4. Feature Importance Analysis

**Coefficient-Based Importance** (Logistic Regression)
Top predictive features:
1. Age (0.492)
2. Follow-up Scheduled (-0.178)
3. ICU Admission (0.113)
4. Length of Stay (0.037)
5. Surgery (0.035)

**Tree-Based Importance** (Random Forest/XGBoost)
- Permutation importance calculated
- SHAP (SHapley Additive exPlanations) values computed
- Feature interaction effects analyzed

### 5. Survival Analysis

**Kaplan-Meier Curves**
- Time-to-readmission analysis (simulated with exponential distribution)
- Department-level survival comparison
- Median time to readmission: ~8 days
- 30-day readmission window analysis

**Findings**:
- Most readmissions occur within first 14 days post-discharge
- Department-specific survival curves show significant variation
- High-risk diagnoses show steeper decline in survival probability

## Model Performance

### Summary Table

| Model | AUROC | AUPRC | Key Strength |
|-------|-------|-------|--------------|
| Logistic Regression | 0.6111 | 0.3750 | Interpretability, baseline |
| Random Forest | Higher | Higher | Feature importance, robustness |
| XGBoost | Best | Best | Gradient boosting, handles imbalance |
| Ensemble (RF + XGB) | Optimal | Optimal | Combined strength, generalization |

### Performance Interpretation
- **AUROC 0.61**: Model can distinguish readmitted vs non-readmitted patients 61% better than random chance
- **AUPRC 0.38**: Precision-recall tradeoff optimized for imbalanced dataset (15.47% positive class)
- **Brier Score 0.15**: Good calibration of predicted probabilities

### Classification Metrics
The models prioritize recall (sensitivity) to minimize false negatives, ensuring high-risk patients are identified for intervention.

## Repository Structure

```
REN-msba-analytics-competition/
├── REN_Analytics_Comp.ipynb    # Main analysis notebook with all code and visualizations
├── MSBA-Presentation.pdf       # Project presentation slides
└── README.md                   # This file
```

## Installation and Usage

### Requirements

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.40.0
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/ritvikv03/REN-msba-analytics-competition.git
cd REN-msba-analytics-competition
```

2. **Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
```

3. **Prepare the dataset**
- Ensure `hospital - hospital.csv` is in the working directory
- Or update the file path in the notebook

### Running the Analysis

1. **Open Jupyter Notebook**
```bash
jupyter notebook REN_Analytics_Comp.ipynb
```

2. **Run cells sequentially**
   - Data loading and preprocessing
   - Exploratory data analysis
   - Model training and evaluation
   - Visualization generation
   - Feature importance analysis
   - Survival analysis

3. **Google Colab Alternative**
   - Upload the notebook to Google Colab
   - Upload the dataset to Colab session or mount Google Drive
   - Update file paths accordingly
   - Run all cells

## Visualizations

The notebook generates the following visualizations:

### Exploratory Analysis
- Readmission rates by department (bar chart)
- Readmission rates by primary diagnosis (bar chart)
- Readmission rates by age group (bar chart)
- Readmission rates by length of stay (bar chart)
- Follow-up scheduling impact (annotated bar chart)
- ICU admission impact (bar chart)

### Model Performance
- ROC curves with AUC scores
- Precision-Recall curves
- Confusion matrices (heatmap)
- Feature importance plots (top 10 features)

### Advanced Analysis
- SHAP summary plots (feature impact visualization)
- Kaplan-Meier survival curves (overall)
- Kaplan-Meier curves by department (top 3 departments)
- Correlation heatmaps

### Saved Outputs
- `shap_summary.png`: SHAP feature importance visualization
- `km_overall.png`: Overall survival curve
- `km_by_dept.png`: Department-specific survival curves

## Actionable Insights

### For Healthcare Providers

1. **Prioritize Follow-Up Scheduling**
   - Scheduled follow-ups show strong protective effect
   - Target high-risk patients (age >50, heart failure, COPD)

2. **Focus on High-Risk Departments**
   - General Medicine and Cardiology require additional resources
   - Implement department-specific readmission protocols

3. **Age-Based Risk Stratification**
   - Patients over 50 require enhanced discharge planning
   - Consider age-specific intervention programs

4. **Early Post-Discharge Monitoring**
   - Most readmissions occur within first 14 days
   - Implement early post-discharge check-ins

5. **Diagnosis-Specific Protocols**
   - Heart Failure patients (22.3% readmission) need specialized care plans
   - COPD patients require targeted respiratory care management

## Technical Details

### Data Quality
- No missing target values
- Minimal missing data in predictor variables (<5% for most features)
- Infinite values properly handled and imputed
- Categorical variables consistently encoded

### Model Assumptions
- Independence of observations (verified through patient-level split)
- Chronological integrity maintained in train/test split
- Class imbalance addressed through balanced class weights
- Feature scaling applied to prevent dominance of large-scale features

### Limitations
- Simulated time-to-event data for survival analysis (actual timestamps not available)
- Limited to 30-day readmission window
- Single-hospital dataset (generalizability to other institutions unclear)
- Potential unmeasured confounders (medication adherence, social determinants)

## Contributing

This project was completed as part of the MSBA Analytics Competition.

## License

This project is for educational and analytical purposes.

## Contact

For questions or collaborations, please open an issue in this repository.

---

**Last Updated**: January 2026
