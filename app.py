import streamlit as st
import joblib
import pandas as pd

st.title("The Churn Prediction")

model = joblib.load("model.joblib")  # now this is the full pipeline

contract_options = ['Month-to-month', 'One year', 'Two year']
payment_options = ['Electronic Check', 'Mailed Check', 'Bank Transfer (automatic)', 'Credit Card (automatic)']

monthly_charges = st.number_input('Monthly Charge', min_value=0.0, max_value=1000.0)
tenure = st.number_input("Tenure in Months", min_value=1, max_value=100, value=1, step=1)
contract = st.selectbox('Contract', contract_options)
payment_method = st.selectbox('Payment Method', payment_options)
age = st.number_input("Age",min_value=18,max_value=150)
number_of_dependents =st.number_input("Number of Dependents",min_value=0,max_value=20)
total_charges = st.number_input("Total Charges", min_value=10,max_value=1000)
number_of_referrals =st.number_input("Number of Referrals",min_value=0,max_value=50)

if st.button("Check for Churn"):
    # Build a dataframe with exactly the same column names as your original dataset (X)
    input_df = pd.DataFrame([{
        "Monthly Charge": monthly_charges,
        "Tenure in Months": tenure,
        "Contract": contract,
        "Payment Method": payment_method,
        "Age":age,
        "Number of Dependents":number_of_dependents,
        "Total Charges":total_charges,
        "Number of Referrals":number_of_referrals
        # Add other required columns with default values if needed
    }])

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer likely to CHURN with probability {prediction_proba:.2f}")
    else:
        st.success(f"✅ Customer likely to STAY with probability {1 - prediction_proba:.2f}")
