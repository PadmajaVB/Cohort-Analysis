import streamlit as st
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times, qth_survival_times


class Predict:
    def __init__(self, data, model, customer):
        self.data = data
        self.model = model
        self.customer = customer

    def predict(self):
        censored_subjects = self.data.loc[self.data['Churn_Yes'] == 0]

        # predict_survival_function() creates the matrix containing a survival probability for each remaining customers
        # 'unconditioned' survival function 'cuz some of these curves will predict churn before the customer's
        # current tenure time row index => tenure period; column_index is the data index where Churn_Yes=0
        unconditioned_sf = self.model.predict_survival_function(censored_subjects)

        # We've to condition the prediction on the basis that the customers were still with us when the data was collected
        # c.name => row number(index) of the data where Churn_Yes=0
        # data.loc[c.name, 'tenure'] => tenure value of specific index(c.name) in original data
        # c.loc[data.loc[c.name, 'tenure']]<=1 always in unconditioned_cf, which may not be true cuz the customers might
        # continue using the platform even after the date of collection of data
        conditioned_sf = unconditioned_sf.apply(lambda c: (c / c.loc[self.data.loc[c.name, 'tenure']]).clip(upper=1))

        plt.clf()
        # investigate individual customers and see how the conditioning has affected their survival over the base line
        subject = self.customer
        unconditioned_sf[subject].plot(ls="--", color="#A60628", label="unconditioned")
        conditioned_sf[subject].plot(color="#A60628",
                                 label="conditioned on $T>34$")  # T>34 indicate that the customer is active even after 58 months

        plt.xlabel('tenure period')
        plt.legend()
        # plot_data = pd.DataFrame()
        # plot_data['unconditioned_sf'] = unconditioned_sf[subject]
        # plot_data['conditioned_sf'] = conditioned_sf[subject]
        # st.line_chart(plot_data)
        st.write("""
        ## Predicting churn 
        **Unconditioned survival curve:** This will predict churn before the customer's current tenure time.\n 
        **Conditioned survival curve:**  This takes into account that customers were still with us when the data was collected, 
        resulting in more relevant prediction.
        """)
        st.pyplot(plt)
        return conditioned_sf

    def predict_50(self, conditioned_sf):
        # Predict the month number where the survival chance of customer is 50%
        # This can also be modified as predictions_50 = qth_survival_times(.50, conditioned_sf),
        # where the percentile can be modified depending on our requirement
        predictions_50 = median_survival_times(conditioned_sf)
        st.write('''
        ### predictions_50
        Predicting the month at which the survival chance of the customer is 50%
        ''')
        st.write(predictions_50[[self.customer]])
        return predictions_50

    def predict_remaining_value(self, predictions_50):
        # Investigate the predicted remeaining value that a customer has for the business
        values = predictions_50.T.join(self.data[['MonthlyCharges', 'tenure']])
        values['RemainingValue'] = values['MonthlyCharges'] * (values[0.5] - values[
            'tenure'])  # With this we can predict which customers might inflict the highest damage to the business
        st.write('''
        **Predicted remaining value** that a customer (with 50% survival chance) has for the business 
        ''')
        st.write(values.loc[[self.customer]])
        return values
