import pandas as pd
import os
import streamlit as st

from preprocessing import data_processing
from model import Model
from predict import Predict
from inference import Inference
import calibration


def file_selector(folder_path='./Data/'):
    global selected_filename
    filenames = os.listdir(folder_path)
    filenames = [file for file in filenames if not file.startswith(".")]
    filenames.insert(0, "Select a file")
    selected_filename = st.selectbox('', filenames)
    if selected_filename == "Select a file":
        selected_filename = 'file_not_selected'
        return selected_filename
    return os.path.join(folder_path, selected_filename)


# TODO: Handle error scenario
def read_data(path, file_type):
    global raw_data
    if selected_filename == 'file_not_selected':
        st.write("Please select a customer data file to proceed")
        return
    if file_type.lower() == 'csv':
        raw_data = pd.read_csv(path)
    elif file_type.lower() == 'excel':
        raw_data = pd.read_excel(path)
    st.write('### First few records')
    st.write(raw_data.head(n=5))
    return raw_data


def preprocessing_data():
    global preprocessed_data
    # print("Enter the file type of the raw data (it has to be either csv or excel)")
    # file_type = input()
    file_type = "csv"
    # print("Enter the file path")
    # path = input()
    filename = "./Data/Telco-Customer-Churn.csv"

    st.write('### Select customer data from Data directory')
    folder_path = './Data/'
    filename = file_selector(folder_path=folder_path)

    if selected_filename == 'file_not_selected':
        st.write("Please select a customer data file to proceed")
        return

    raw_data = read_data(filename, file_type)
    preprocessed_data = data_processing(raw_data)


def model_and_visualize():
    global model
    global train_data
    global test_data
    if selected_filename == 'file_not_selected':
        st.write("Please select a customer data file to proceed")
        return
    build_model = Model(preprocessed_data)
    train_data, test_data, model = build_model.coxPH()


def predict():
    global values
    global customer
    global percentile

    if selected_filename == 'file_not_selected':
        st.write("Please select a customer data file to proceed")
        return

    customer = st.sidebar.selectbox(
        "Select a customer for predicting churn",
        ("5575-GNVDE", "1452-KIOVK", "9763-GRSKD")
    )
    risk_factor = st.sidebar.slider(
        "How critical retaining the customer is?",
        0.0, 1.0, 0.50, 0.1
    )
    percentile = risk_factor
    prediction = Predict(train_data, model, customer)
    conditioned_sf = prediction.predict()
    predictions = prediction.predict_percentile(conditioned_sf, percentile)
    values = prediction.predict_remaining_value(predictions, percentile)


def inference():
    global actions
    if selected_filename == 'file_not_selected':
        st.write("### Customer Data")
        return
    inference = Inference(train_data, model, customer)
    actions = inference.churn_prevention(values,percentile)
    actions = inference.financial_impact(actions)


def score():
    if selected_filename == 'file_not_selected':
        st.write("Please select a customer data file to proceed")
        return
    loss = calibration.calibration_plot(model, test_data)
    roi = calibration.return_on_investment(loss,actions)
