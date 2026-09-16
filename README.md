
# Predicting 30-Day Hospital Readmissions in India

## 📌 Project Overview
Hospital readmissions are a major burden on healthcare systems, indicating potential gaps in care and leading to high out-of-pocket costs for patients. This project builds an end-to-end machine learning pipeline to predict the probability of a patient returning to the hospital within 30 days of discharge. 

By identifying high-risk patients early, hospitals can allocate resources (such as social worker follow-ups or specialized discharge nursing) more effectively.

## 🎯 Key Achievements
* **Engineered Clinical Features:** Created custom interaction variables combining Length of Stay (LOS) and the Charlson Comorbidity Index to capture complex clinical realities.
* **Addressed Class Imbalance:** Successfully managed an ~11.8% readmission rate baseline using SMOTE and targeted class weighting.
* **Optimized Performance:** Achieved an **ROC-AUC score of 0.7622** by tuning hyperparameters using `RandomizedSearchCV`.
* **Model Explainability:** Implemented **SHAP (SHapley Additive exPlanations)** to transition the XGBoost model from a "black box" to a transparent clinical decision-support tool.

## 📊 Business & Clinical Insights
Using SHAP analysis, the model revealed the top drivers of hospital readmissions:
1. **Clinical Risk Interaction:** The engineered feature `risk_los_x_charlson` emerged as a top predictor, showing that sick patients (high Charlson index) who experience prolonged hospital stays have an exponentially higher risk of bouncing back.
2. **Discharge Behavior:** Patients discharged as "LAMA" (Left Against Medical Advice) show a massive spike in readmission probability, proving that behavioral/administrative statuses are often stronger predictors than physiological lab results.
3. **Socioeconomic Factors:** The presence of a BPL (Below Poverty Line) card combined with medical complexity showed a measurable increase in risk, highlighting the need for post-discharge social work.

## ⚙️ Technical Stack
* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `category_encoders`, `imbalanced-learn`
* **Model Explainability:** `shap`
* **Visualization:** `matplotlib`

## 📁 Dataset
This project uses the synthetic **India Hospital Readmission Dataset (2015–2024)** sourced from Kaggle. The dataset mimics real-world relational hospital databases across five tables:
* `admissions.csv`
* `patients.csv`
* `diagnoses.csv`
* `hospitals.csv`
* `billing.csv`

## 🚀 How to Run the Notebook
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed (`pip install -r requirements.txt`). *(Note: you may need to create a requirements.txt file or list the libraries here)*.
3. Download the dataset from [Kaggle Link Here] and place the 5 CSV files in the same directory as the notebook.
4. Run `Predicting_30_Day_Hospital_Readmissions_in_India.ipynb` from top to bottom.

---
*Developed as a demonstration of clinical machine learning, feature engineering, and model transparency.*
