import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

class Inference:
    def __init__(self, train_data, model, customer):
        self.data = train_data
        self.model = model
        self.customer = customer

    def churn_prevention(self, values,percentile):
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
        actions = values.join(result_df).drop([percentile], axis=1)
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

