import streamlit as st
import pandas as pd
import joblib

st.title(" Hospital Readmission Risk Predictor")
st.write("Enter patient details to predict 30-day readmission risk.")

# Load the saved model and columns
model = joblib.load('xgboost_readmission_model.pkl')
columns = joblib.load('model_columns.pkl')

# Create inputs for the user
los_days = st.number_input("Length of Stay (days)", min_value=1, value=4)
charlson_index = st.number_input("Charlson Comorbidity Index", min_value=0, value=2)
is_lama = st.checkbox("Discharged Against Medical Advice (LAMA)?")

if st.button("Predict Risk"):
    # Create an empty dataframe with all expected columns set to 0
    input_data = pd.DataFrame(0, index=[0], columns=columns)
    
    # Update the features based on user input
    if 'los_days' in input_data.columns: 
        input_data['los_days'] = los_days
    if 'charlson_index' in input_data.columns: 
        input_data['charlson_index'] = charlson_index
    if 'risk_los_x_charlson' in input_data.columns: 
        input_data['risk_los_x_charlson'] = los_days * charlson_index
    if 'discharge_type_LAMA' in input_data.columns and is_lama: 
        input_data['discharge_type_LAMA'] = 1
    
    # Generate prediction
    probability = model.predict_proba(input_data)[0][1]
    
    # Display result
    if probability > 0.15:
        st.error(f"High Risk of Readmission: {probability:.1%}")
    else:
        st.success(f"Low Risk of Readmission: {probability:.1%}")
