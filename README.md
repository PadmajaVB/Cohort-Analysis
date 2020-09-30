# Cohort-Analysis

Cohort analysis is performed when you want to understand how the group of users are behaving (also known as quantitative analysis). 

In this repository we are performing cohort analysis on simple retail data set and you can access the code from [CustomerRetention.ipynb](https://github.com/PadmajaVB/Cohort-Analysis/blob/master/Code/CustomerRetention.ipynb) 


# Churn-Analysis

This is performed when you want to understand what exact features is causing your customers to churn (also known as qualitative analyis).

Here we are building a very cool streamlit web application that can be easily deployed on Heroku. 

### Command to run the streamlit application on local server
`streamlit run ./Code/app.py`

### Steps to deploy the application on heroku

1. Make sure the requirement.txt has all the required depedencies mentioned. 
2. Create a proc file that has the commands to run the streamlit application once the app is deployed on cloud. 
3. Run these commands to deploy the code on heroku

    i. `git add .`
    
    ii. `git commit -m "deploying streamlit app on heroku`
    
    iii. `git push heroku master`
    
    iv. `heroku ps:scale web=1`
    
You can access our heroku appliction for churn analysis from here - https://churn-analysis.herokuapp.com/
