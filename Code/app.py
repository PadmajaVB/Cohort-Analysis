from collections import OrderedDict

import streamlit as st

import main


PAGES = OrderedDict(
    [
        ("Data", (main.preprocessing_data, None)),

        ("Model", (main.model_and_visualize, None)),
        (
            "Prediction", (main.predict, None)
        ),
        (
            "What If Analysis",
            (
                main.inference,
                """Here we will see how altering any predictor impacts the survival chances of the customer""",
            ),
        ),
        (
            "Model Diagnostics",
            (
                main.score,
                """Here we will be calculating the accuracy and calibration of our model.
                 \n Calibration is the propensity of the model to get probabilities right over time (i.e. having high recall value)""",
            ),
        )
    ]
)


def main():
    demo_name = st.sidebar.selectbox("Quick links", list(PAGES.keys()), 0)
    demo = PAGES[demo_name][0]

    if demo_name == "Data":
        st.write("# Churn Analysis")
    elif demo_name == "Model":
        st.write('# Machine learning models are built to predict churn')
        st.write('(Details at the end of the page)')
    elif demo_name == "Prediction":
        st.write('# Propensity to Churn')
        st.write('Here we will be predicting the survival probability and remaining value (monetary value) '
                 'of the selected customer.')
    else:
        st.markdown("# %s" % demo_name)
        description = PAGES[demo_name][1]
        if description:
            st.write(description)
        # Clear everything from the intro page.
        # We only have 4 elements in the page so this is intentional overkill.
        for i in range(10):
            st.empty()

    demo()


if __name__ == "__main__":
    main()
