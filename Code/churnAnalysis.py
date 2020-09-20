import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times, qth_survival_times
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

pd.set_option('display.max_columns', 50)

"""
# Churn Analysis 
Analysing at-risk customers, factors that influence the churn, and ways to prevent it.
"""


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


def data_processing(raw_data):
    dummies = pd.get_dummies(
        raw_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']]
    )
    dummies = dummies[['gender_Female', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
                       'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
                       'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
                       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                       'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                       'PaymentMethod_Electronic check', 'Churn_Yes']]
    data = dummies.join(raw_data[['customerID', 'MonthlyCharges', 'TotalCharges', 'tenure']])
    data.set_index('customerID', inplace=True)
    data['TotalCharges'] = data[['TotalCharges']].replace([' '], '0')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
    if st.checkbox('Show processed data'):
        st.write(data.head(n=5))
    return data


def modeling(data):
    train_features = ['gender_Female', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
                      'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes', 'DeviceProtection_Yes',
                      'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
                      'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                      'PaymentMethod_Electronic check','MonthlyCharges', 'TotalCharges','tenure', 'Churn_Yes']

    cph_train, cph_test = train_test_split(data[train_features], test_size=0.2)
    cph = CoxPHFitter()
    cph.fit(cph_train, 'tenure', 'Churn_Yes')
    return cph_train, cph_test, cph


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


def predict(data, model):
    # censored observation is one which is yet to have an ‘event’, i.e. customers who are yet to churn.
    censored_subjects = data.loc[data['Churn_Yes'] == 0]

    # predict_survival_function() creates the matrix containing a survival probability for each remaining customers
    # 'unconditioned' survival function 'cuz some of these curves will predict churn before the customer's current tenure time
    # row index => tenure period; column_index is the data index where Churn_Yes=0
    unconditioned_sf = model.predict_survival_function(censored_subjects)

    # We've to condition the prediction on the basis that the customers were still with us when the data was collected
    # c.name => row number(index) of the data where Churn_Yes=0
    # data.loc[c.name, 'tenure'] => tenure value of specific index(c.name) in original data
    # c.loc[data.loc[c.name, 'tenure']]<=1 always in unconditioned_cf, which may not be true cuz the customers might
    # continue using the platform even after the date of collection of data
    conditioned_sf = unconditioned_sf.apply(lambda c: (c / c.loc[data.loc[c.name, 'tenure']]).clip(upper=1))

    plt.clf()
    # investigate individual customers and see how the conditioning has affected their survival over the base line
    subject = '5575-GNVDE'
    unconditioned_sf[subject].plot(ls="--", color="#A60628", label="unconditioned")
    conditioned_sf[subject].plot(color="#A60628",
                                 label="conditioned on $T>34$")  # T>34 indicate that the customer is active even after 58 months

    plt.legend()
    # plot_data = pd.DataFrame()
    # plot_data['unconditioned_sf'] = unconditioned_sf[subject]
    # plot_data['conditioned_sf'] = conditioned_sf[subject]
    # st.line_chart(plot_data)
    '''
    ## Predicting churn 
    **Unconditioned survival curve:** This will predict churn before the customer's current tenure time.\n 
    **Conditioned survival curve:**  This takes into account that customers were still with us when the data was collected, 
    resulting in more relevant prediction.
    '''
    st.pyplot(plt)
    return conditioned_sf


def predict_50(conditioned_sf):
    # Predict the month number where the survival chance of customer is 50%
    # This can also be modified as predictions_50 = qth_survival_times(.50, conditioned_sf),
    # where the percentile can be modified depending on our requirement
    predictions_50 = median_survival_times(conditioned_sf)
    return predictions_50


def predict_value(predictions_50):
    # Investigate the predicted remeaining value that a customer has for the business
    values = predictions_50.T.join(data[['MonthlyCharges', 'tenure']])
    values['RemainingValue'] = values['MonthlyCharges'] * (values[0.5] - values[
        'tenure'])  # With this we can predict which customers might inflict the highest damage to the business
    '''
    **Predicted remaining value** that a customer (with 50% survival chance) has for the business 
    '''
    st.write(values.head(n=5))
    return values


def churn_prevention(values, model):
    # Through coefficient chart we concluded that these 4 features i.e.
    # Contract_Two year, Contract_One year, PaymentMethod_Credit card(automatic), PaymentMethod_Bank transfer(automatic)
    # promotes the survival chances positively, so let's focus on those

    upgrades = ['PaymentMethod_Credit card (automatic)', 'PaymentMethod_Bank transfer (automatic)', 'Contract_One year',
                'Contract_Two year']
    results_dict = {}

    # TODO: run this for all the customers
    actual = data.loc[['5575-GNVDE']]
    change = data.loc[['5575-GNVDE']]
    results_dict['5575-GNVDE'] = [model.predict_median(actual)]
    for upgrade in upgrades:
        change[upgrade] = 1 if list(change[upgrade]) == [0] else 0
        results_dict['5575-GNVDE'].append(model.predict_median(change))
        change[upgrade] = 1 if list(change[upgrade]) == [0] else 0

    result_df = pd.DataFrame(results_dict).T
    result_df.columns = ['baseline'] + upgrades
    actions = values.join(result_df).drop([0.5], axis=1)
    # Notice that if we get the 1st customer to use credit card, we increase the survival period by 4 months i.e.
    # 22(baseline) -> 26(PaymentMethod_Credit card (automatic)) and so on..
    print("Change in survival period\n", actions.head())
    '''
    ## Churn Prevention
    Reference data to see what features the customer had already subscribed/unsubscribed to.
    '''
    st.write(data.loc[['5575-GNVDE'], upgrades])
    '''
    \n
    Through coefficient chart we concluded that these 4 features i.e. *Contract_Two year, Contract_One year, 
    PaymentMethod_Credit card (automatic), PaymentMethod_Bank transfer (automatic)* promotes the survival chances positively, 
    so let's focus on those and see how subscribing/ unsubscribing to these services changes the survival chances
    '''
    st.write(actions[['baseline', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Bank transfer (automatic)',
                     'Contract_One year', 'Contract_Two year']].head(n=5))
    '''
    \n
    Notice that if we get the 1st customer to use CC we increase the survival period of cust '5575-GNVDE' by 5 months 
    i.e. 46(baseline) -> 51(PaymentMethod_Credit card (automatic)) and so on..
    Note: Cust 5575-GNVDE was already having Contract_One year, after reverting it we can see that the survival chances 
    goes down from 46 to 37
    '''
    st.write(actions.loc[['5575-GNVDE']].head(n=5))

    return actions


def financial_impact(actions):
    actions['CreditCard Diff'] = (actions['PaymentMethod_Credit card (automatic)'] - actions['baseline']) * actions[
        'MonthlyCharges']
    actions['BankTransfer Diff'] = (actions['PaymentMethod_Bank transfer (automatic)'] - actions['baseline']) * actions[
        'MonthlyCharges']
    actions['1yrContract Diff'] = (actions['Contract_One year'] - actions['baseline']) * actions['MonthlyCharges']
    actions['2yrContract Diff'] = (actions['Contract_Two year'] - actions['baseline']) * actions['MonthlyCharges']
    print("Financial impact that the change in survival period has:\n", actions.head())
    '''
    \n
    **Financial Impact:** This tells us the additional financial value that the customer can add by subscribing 
    to each on of the given four features
    '''
    st.write(actions.loc[['5575-GNVDE']])
    return actions


def calibration_plot(test_data, model):
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

    '''
    ## Accuracy and Calibration
    Calibration is the propensity of the model to get probabilities right over time (i.e. having high recall value)
    
    **Calibration plot** for tenure = 13 months 
    '''
    st.pyplot(plt)

    # To understand how far away the line is from the perfect calibration we use brier_score_loss
    brier_score_loss(test_data['Churn_Yes'], 1 - np.array(model.predict_survival_function(test_data).loc[13]), pos_label=1)

    # Inspect the calibration of the model at all the time periods (above one is just for tenure=13)
    loss_dict = {}
    for i in range(1, 73):
        score = brier_score_loss(test_data['Churn_Yes'], 1 - np.array(model.predict_survival_function(test_data).loc[i]),
                                 pos_label=1)
        loss_dict[i] = [score]

    loss_df = pd.DataFrame(loss_dict).T

    fig, ax = plt.subplots()
    ax.plot(loss_df.index, loss_df)
    ax.set(xlabel='Prediction Time', ylabel='Calibration Loss', title='Cox PH Model Calibration Loss / Time')
    ax.grid()

    # Here we can see that the model is well caliberated b/w 5 and 25 months
    plt.show()

    '''
    **Brier score loss plot** after inspecting the calibration of model for all the tenures/time period
    '''
    st.pyplot(plt)

    return loss_df


def return_on_investment(loss_df):
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


if __name__ == "__main__":
    # print("Enter the file type of the raw data (it has to be either csv or excel)")
    # file_type = input()
    file_type = "csv"
    # print("Enter the file path")
    # path = input()
    path = "/Users/pbhagwat/DEV/CohortAnalysis/Cohort-Analysis/Data/Telco-Customer-Churn.csv"

    raw_data = read_data(path, file_type)
    data = data_processing(raw_data)
    train_data, test_data, model = modeling(data)
    visualize(model)
    conditioned_sf = predict(train_data, model)
    predictions_50 = predict_50(conditioned_sf)
    values = predict_value(predictions_50)
    actions = churn_prevention(values, model)
    actions = financial_impact(actions)
    loss_df = calibration_plot(test_data,model)
    actions = return_on_investment(loss_df)

