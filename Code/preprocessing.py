import pandas as pd
import streamlit as st

def data_processing(raw_data):
    dummies = pd.get_dummies(
        raw_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']]
    )
    dummies = dummies[['SeniorCitizen','gender_Female', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
                       'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
                       'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
                       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                       'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                       'PaymentMethod_Electronic check', 'Churn_Yes']]
    data = dummies.join(raw_data[['customerID', 'MonthlyCharges', 'TotalCharges', 'tenure']])
    data.set_index('customerID', inplace=True)
    data['TotalCharges'] = data[['TotalCharges']].replace([' '], '0')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
    if st.checkbox('Show pre-processed data'):
        st.write(data.head(n=5))
    return data

