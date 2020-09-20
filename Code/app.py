from collections import OrderedDict

import streamlit as st
import awesome_streamlit as ast

import main

ast.core.services.other.set_logging_format()

PAGES = OrderedDict(
    [
        ("Data", (main.preprocessing_data, None)),

        ("Model", (main.model_and_visualize, None)),
        (
            "Prediction",
            (
                main.predict,
                """
                Here we will be predicting the survival probability and remaining value (monetary value)
                 of the selected customer.""",
            ),
        ),
        (
            "Inference",
            (
                main.inference,
                """Here we will see how altering any predictor impacts the survival chances of the customer""",
            ),
        )
    ]
)


def main():
    demo_name = st.sidebar.selectbox("Navigation menu", list(PAGES.keys()), 0)
    demo = PAGES[demo_name][0]

    if demo_name == "Data":
        st.write("# Churn Analysis")
        st.write("""This help in analysing at-risk customers, factors that influence the churn, 
                and ways to prevent it.\n Here we will be pre-processing the data""")
    elif demo_name == "Model":
        st.write('# Using CoxPH model for survival analysis')
        st.write("""Here we will be using one of the survial model called as CoXPH for predicting churn""")
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
