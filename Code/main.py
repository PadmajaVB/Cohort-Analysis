import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
    if st.checkbox('Show raw data'):
        st.write(raw_data.head(n=5))
    return raw_data


def visualize(model):
    plt.clf()
    model.print_summary()
    '''
    ## Using CoxPH model for survival analysis
    **Feature significance chart aka coefficient chart:** This tells us the relative significance of each feature on the 
    customer churn. 
    Feature with positive coef increases the probability of customer churn and feature with negative coef 
    reduces the churn probability.
    '''
    model.plot()
    st.pyplot(plt)
    '''
    \n
    **Survival curves** for customers whose TotalCharges are 4000, 2500, 2000 and 0. 
    Clearly customers with high TotalCharges have high survival chances. 
    '''
    model.plot_covariate_groups('TotalCharges', [0, 2000, 2500, 4000], cmap='coolwarm')
    st.pyplot(plt)


if __name__ == "__main__":
    # print("Enter the file type of the raw data (it has to be either csv or excel)")
    # file_type = input()
    file_type = "csv"
    # print("Enter the file path")
    # path = input()
    path = "../Data/Telco-Customer-Churn.csv"

    raw_data = read_data(path, file_type)
    data = data_processing(raw_data)

    model = Model(data)
    train_data, test_data, model = model.coxPH()

    visualize(model)

    prediction = Predict(data, model)
    conditioned_sf = prediction.predict()
    predictions_50 = prediction.predict_50(conditioned_sf)
    values = prediction.predict_remaining_value(predictions_50)

    inference = Inference(data, model)
    actions = inference.churn_prevention(values)
    actions = inference.financial_impact(actions)
    loss = inference.calibration_plot(test_data)
    actions = inference.return_on_investment(loss, actions)