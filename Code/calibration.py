import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_plot(model, test_data):
    plt.clf()
    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    probs = 1 - np.array(model.predict_survival_function(test_data).loc[13])  # here tenure=13

    actual = test_data['Churn_Yes']

    fraction_of_positives, mean_predicted_value = calibration_curve(actual, probs, n_bins=10, normalize=False)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % ("CoxPH"))

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel("mean_predicted_value")
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')

    st.write('''
    **Calibration plot** for tenure = 13 months 
    ''')
    st.pyplot(plt)

    # To understand how far away the line is from the perfect calibration we use brier_score_loss
    brier_score_loss(test_data['Churn_Yes'], 1 - np.array(model.predict_survival_function(test_data).loc[13]),
                     pos_label=1)

    # Inspect the calibration of the model at all the time periods (above one is just for tenure=13)
    loss_dict = {}
    for i in range(1, 73):
        score = brier_score_loss(test_data['Churn_Yes'],
                                 1 - np.array(model.predict_survival_function(test_data).loc[i]),
                                 pos_label=1)
        loss_dict[i] = [score]

    loss_df = pd.DataFrame(loss_dict).T

    fig, ax = plt.subplots()
    ax.plot(loss_df.index, loss_df.values)
    ax.set(xlabel='Prediction Time', ylabel='Calibration Loss', title='Cox PH Model Calibration Loss / Time')
    ax.grid()

    # Here we can see that the model is well caliberated b/w 5 and 25 months
    plt.show()

    st.write('''
    **Brier score loss plot** after inspecting the calibration of model for all the tenures/time period
    ''')
    st.pyplot(plt)

    return loss_df


def return_on_investment(loss_df, actions):
    # upper and lower bounds for the expected return on investment from getting customers to make changes

    loss_df.columns = ['loss']

    temp_df = actions.reset_index().set_index('PaymentMethod_Credit card (automatic)').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['CreditCard Lower'] = temp_df['CreditCard Diff'] - (temp_df['loss'] * temp_df['CreditCard Diff'])
    actions['CreditCard Upper'] = temp_df['CreditCard Diff'] + (temp_df['loss'] * temp_df['CreditCard Diff'])

    temp_df = actions.reset_index().set_index('PaymentMethod_Bank transfer (automatic)').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['BankTransfer Lower'] = temp_df['BankTransfer Diff'] - (.5 * temp_df['loss'] * temp_df['BankTransfer Diff'])
    actions['BankTransfer Upper'] = temp_df['BankTransfer Diff'] + (.5 * temp_df['loss'] * temp_df['BankTransfer Diff'])

    temp_df = actions.reset_index().set_index('Contract_One year').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['1yrContract Lower'] = temp_df['1yrContract Diff'] - (.5 * temp_df['loss'] * temp_df['1yrContract Diff'])
    actions['1yrContract Upper'] = temp_df['1yrContract Diff'] + (.5 * temp_df['loss'] * temp_df['1yrContract Diff'])

    temp_df = actions.reset_index().set_index('Contract_Two year').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['2yrContract Lower'] = temp_df['2yrContract Diff'] - (.5 * temp_df['loss'] * temp_df['2yrContract Diff'])
    actions['2yrContract Upper'] = temp_df['2yrContract Diff'] + (.5 * temp_df['loss'] * temp_df['2yrContract Diff'])

    return actions
