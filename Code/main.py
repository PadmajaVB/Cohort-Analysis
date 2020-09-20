import pandas as pd
import streamlit as st

from preprocessing import data_processing
from model import Model
from predict import Predict
from inference import Inference


# TODO: Handle error scenario
def read_data(path, file_type):
    global raw_data
    if file_type.lower() == 'csv':
        raw_data = pd.read_csv(path)
    elif file_type.lower() == 'excel':
        raw_data = pd.read_excel(path)
    if st.checkbox('Show sample raw data'):
        st.write(raw_data.head(n=5))
    return raw_data


def preprocessing_data():
    global preprocessed_data
    # print("Enter the file type of the raw data (it has to be either csv or excel)")
    # file_type = input()
    file_type = "csv"
    # print("Enter the file path")
    # path = input()
    path = "./Data/Telco-Customer-Churn.csv"

    raw_data = read_data(path, file_type)
    preprocessed_data = data_processing(raw_data)


def model_and_visualize():
    global model, train_data, test_data
    build_model = Model(preprocessed_data)
    train_data, test_data, model = build_model.coxPH()


def predict():
    global values
    global customer

    customer = st.sidebar.selectbox(
        "Select a customer for predicting churn",
        ("5575-GNVDE", "1452-KIOVK", "9763-GRSKD")
    )
    prediction = Predict(train_data, model, customer)
    conditioned_sf = prediction.predict()
    predictions_50 = prediction.predict_50(conditioned_sf)
    values = prediction.predict_remaining_value(predictions_50)


def inference():
    inference = Inference(train_data, model, customer)
    actions = inference.churn_prevention(values)
    actions = inference.financial_impact(actions)
    loss = inference.calibration_plot(test_data)
    actions = inference.return_on_investment(loss, actions)
