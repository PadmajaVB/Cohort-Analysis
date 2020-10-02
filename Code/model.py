from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import streamlit as st


class Model:
    def __init__(self, data):
        self.data = data

    def coxPH(self):
        train_features = ['gender_Female', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
                          'InternetService_DSL', 'InternetService_Fiber optic', 'OnlineSecurity_Yes',
                          'DeviceProtection_Yes',
                          'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year',
                          'Contract_Two year',
                          'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)',
                          'PaymentMethod_Credit card (automatic)',
                          'PaymentMethod_Electronic check', 'MonthlyCharges', 'TotalCharges', 'tenure', 'Churn_Yes']

        cph_train, cph_test = train_test_split(self.data[train_features], test_size=0.2, random_state=100)
        cph = CoxPHFitter()
        cph.fit(cph_train, 'tenure', 'Churn_Yes')
        self.visualize(cph)
        return cph_train, cph_test, cph

    def visualize(self, model):
        plt.clf()
        model.print_summary()
        st.write('''
        **Feature significance chart aka coefficient chart:** This tells us the relative significance of each feature on the 
        customer churn. 
        Feature with positive coef increases the probability of customer churn and feature with negative coef 
        reduces the churn probability.
        ''')
        model.plot()
        st.pyplot(plt)
        st.write('''
        \n
        **Survival curves** for customers whose TotalCharges are 4000, 2500, 2000 and 0. 
        Clearly customers with high TotalCharges have high survival chances. 
        ''')
        model.plot_partial_effects_on_outcome('TotalCharges', [0, 2000, 2500, 4000], cmap='coolwarm').set_xlabel('tenure period')
        st.pyplot(plt)

        st.write('### Approach')
        st.write("""Survival analysis models are used to predict churn. 
        It helps you predict the survival chances of the customer at any given point of time.
        Here we have used one type of survival analysis model called as CoXPH for predicting churn""")
        link = '[Read more](https://en.wikipedia.org/wiki/Survival_analysis)'
        st.markdown(link, unsafe_allow_html=True)
