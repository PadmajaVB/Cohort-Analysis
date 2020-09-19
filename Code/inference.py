import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class Inference:
    def __init__(self, train_data, model, customer):
        self.data = train_data
        self.model = model
        self.customer = customer

    def churn_prevention(self, values):
        # Through coefficient chart we concluded that these 4 features i.e.
        # Contract_Two year, Contract_One year, PaymentMethod_Credit card(automatic), PaymentMethod_Bank transfer(automatic)
        # promotes the survival chances positively, so let's focus on those

        upgrades = ['PaymentMethod_Credit card (automatic)', 'PaymentMethod_Bank transfer (automatic)',
                    'Contract_One year',
                    'Contract_Two year']
        results_dict = {}

        st.write('''
        ### Original Subscription
        ''')
        st.write(self.data.loc[[self.customer], upgrades].T)

        # TODO: run this for all the customers
        actual = self.data.loc[[self.customer]]
        change = self.data.loc[[self.customer]]
        results_dict[self.customer] = [self.model.predict_median(actual)]
        for upgrade in upgrades:
            change[upgrade] = 1 if list(change[upgrade]) == [0] else 0
            results_dict[self.customer].append(self.model.predict_median(change))
            change[upgrade] = 1 if list(change[upgrade]) == [0] else 0

        result_df = pd.DataFrame(results_dict).T
        result_df.columns = ['baseline'] + upgrades
        actions = values.join(result_df).drop([0.5], axis=1)
        # Notice that if we get the 1st customer to use credit card, we increase the survival period by 4 months i.e.
        # 22(baseline) -> 26(PaymentMethod_Credit card (automatic)) and so on..
        print("Change in survival period\n", actions.head())
        st.write('''
        ## Churn Prevention
        Reference data to see what features the customer had already subscribed/unsubscribed to.
        ''')
        st.write(self.data.loc[[self.customer], upgrades])
        st.write('''
        \n
        Through coefficient chart we concluded that these 4 features i.e. *Contract_Two year, Contract_One year, 
        PaymentMethod_Credit card (automatic), PaymentMethod_Bank transfer (automatic)* promotes the survival chances positively, 
        so let's focus on those and see how subscribing/ unsubscribing to these services changes the survival chances
        ''')
        source = actions.loc[[self.customer], upgrades].T
        source['upgrades'] = actions.loc[[self.customer], upgrades].T.index
        source['baseline'] = actions.loc[self.customer]['baseline']
        source = source.rename(columns={self.customer: 'Survival period(months)'})

        bar = alt.Chart(source).mark_bar(size=50).encode(
            x=alt.X('upgrades:O', sort=upgrades),
            y='Survival period(months):Q'
        )

        rule = alt.Chart(source).mark_rule(color='red').encode(
            y='baseline:Q',
        )

        custom_chart = (bar + rule).properties(height=700, width=700)
        st.altair_chart(custom_chart)

        return actions

    def financial_impact(self, actions):
        actions['CreditCard Diff'] = (actions['PaymentMethod_Credit card (automatic)'] - actions['baseline']) * actions[
            'MonthlyCharges']
        actions['BankTransfer Diff'] = (actions['PaymentMethod_Bank transfer (automatic)'] - actions['baseline']) * \
                                       actions[
                                           'MonthlyCharges']
        actions['1yrContract Diff'] = (actions['Contract_One year'] - actions['baseline']) * actions['MonthlyCharges']
        actions['2yrContract Diff'] = (actions['Contract_Two year'] - actions['baseline']) * actions['MonthlyCharges']
        print("Financial impact that the change in survival period has:\n", actions.head())
        st.write('''
        \n
        **Financial Impact:** This tells us the additional financial value that the customer can add by subscribing 
        to each on of the given four features
        ''')
        upgrades = ['CreditCard Diff', 'BankTransfer Diff', '1yrContract Diff', '2yrContract Diff']
        source = actions.loc[[self.customer], upgrades].T
        source = source.rename(columns={self.customer: 'Monetary value'})
        source['Monetary value'] = actions.loc[self.customer]['RemainingValue'] + source['Monetary value']
        source['upgrades'] = actions.loc[[self.customer], upgrades].T.index
        source['baseline'] = actions.loc[self.customer]['RemainingValue']

        bar = alt.Chart(source).mark_bar(size=50).encode(
            x=alt.X('upgrades:O', sort=upgrades),
            y='Monetary value:Q'
        )

        rule = alt.Chart(source).mark_rule(color='red').encode(
            y='baseline:Q'
        )

        custom_chart = (bar + rule).properties(height=700, width=700)
        st.altair_chart(custom_chart)

        return actions

    def calibration_plot(self, test_data):
        plt.clf()
        plt.figure(figsize=(5, 5))

        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        probs = 1 - np.array(self.model.predict_survival_function(test_data).loc[13])  # here tenure=13

        actual = test_data['Churn_Yes']

        fraction_of_positives, mean_predicted_value = calibration_curve(actual, probs, n_bins=10, normalize=False)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % ("CoxPH"))

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlabel("mean_predicted_value")
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots (reliability curve)')

        st.write('''
        ## Accuracy and Calibration
        Calibration is the propensity of the model to get probabilities right over time (i.e. having high recall value)

        **Calibration plot** for tenure = 13 months 
        ''')
        st.pyplot(plt)

        # To understand how far away the line is from the perfect calibration we use brier_score_loss
        brier_score_loss(test_data['Churn_Yes'], 1 - np.array(self.model.predict_survival_function(test_data).loc[13]),
                         pos_label=1)

        # Inspect the calibration of the model at all the time periods (above one is just for tenure=13)
        loss_dict = {}
        for i in range(1, 73):
            score = brier_score_loss(test_data['Churn_Yes'],
                                     1 - np.array(self.model.predict_survival_function(test_data).loc[i]),
                                     pos_label=1)
            loss_dict[i] = [score]

        loss_df = pd.DataFrame(loss_dict).T

        fig, ax = plt.subplots()
        ax.plot(loss_df.index, loss_df)
        ax.set(xlabel='Prediction Time', ylabel='Calibration Loss', title='Cox PH Model Calibration Loss / Time')
        ax.grid()

        # Here we can see that the model is well caliberated b/w 5 and 25 months
        plt.show()

        st.write('''
        **Brier score loss plot** after inspecting the calibration of model for all the tenures/time period
        ''')
        st.pyplot(plt)

        return loss_df

    def return_on_investment(self, loss_df, actions):
        # upper and lower bounds for the expected return on investment from getting customers to make changes

        loss_df.columns = ['loss']

        temp_df = actions.reset_index().set_index('PaymentMethod_Credit card (automatic)').join(loss_df)
        temp_df = temp_df.set_index('index')
        actions['CreditCard Lower'] = temp_df['CreditCard Diff'] - (temp_df['loss'] * temp_df['CreditCard Diff'])
        actions['CreditCard Upper'] = temp_df['CreditCard Diff'] + (temp_df['loss'] * temp_df['CreditCard Diff'])

        temp_df = actions.reset_index().set_index('PaymentMethod_Bank transfer (automatic)').join(loss_df)
        temp_df = temp_df.set_index('index')
        actions['BankTransfer Lower'] = temp_df['BankTransfer Diff'] - (
                    .5 * temp_df['loss'] * temp_df['BankTransfer Diff'])
        actions['BankTransfer Upper'] = temp_df['BankTransfer Diff'] + (
                    .5 * temp_df['loss'] * temp_df['BankTransfer Diff'])

        temp_df = actions.reset_index().set_index('Contract_One year').join(loss_df)
        temp_df = temp_df.set_index('index')
        actions['1yrContract Lower'] = temp_df['1yrContract Diff'] - (
                    .5 * temp_df['loss'] * temp_df['1yrContract Diff'])
        actions['1yrContract Upper'] = temp_df['1yrContract Diff'] + (
                    .5 * temp_df['loss'] * temp_df['1yrContract Diff'])

        temp_df = actions.reset_index().set_index('Contract_Two year').join(loss_df)
        temp_df = temp_df.set_index('index')
        actions['2yrContract Lower'] = temp_df['2yrContract Diff'] - (
                    .5 * temp_df['loss'] * temp_df['2yrContract Diff'])
        actions['2yrContract Upper'] = temp_df['2yrContract Diff'] + (
                    .5 * temp_df['loss'] * temp_df['2yrContract Diff'])

        return actions