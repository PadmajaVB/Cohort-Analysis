from sklearn.cross_validation import train_test_split
from lifelines import CoxPHFitter


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

        cph_train, cph_test = train_test_split(self.data[train_features], test_size=0.2)
        cph = CoxPHFitter()
        cph.fit(cph_train, 'tenure', 'Churn_Yes')
        return cph_train, cph_test, cph

