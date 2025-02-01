import streamlit as st
import requests
import os

# Streamlit interface
st.markdown("<h1 style='text-align: center;'>Backorder Prediction</h1>", unsafe_allow_html=True)

st.write('Enter the required details to predict backorder situation')

# Create form to take user input
inv = st.number_input('Current National Inventory of the product', value= 0.0, step=1.0)
leadtime = st.number_input('Lead Time of the product', min_value=0.0, step=0.1)
transit = st.number_input('Total quantity of products in transit', min_value=0.0, step=1.0)
forecast = st.number_input('Forecasted Sales of the product for next 3 months', min_value=0.0, step=1.0)
sales = st.number_input('Previous month Sales of the product', min_value=0.0, step=1.0)
minbank = st.number_input('Minimum Quantity of Products required to keep in stock', min_value=0.0, step=1.0)
issue = st.radio('Is there a potential issue with the product?', ['Yes','No'])
piecesdue = st.number_input('No of pieces/products overdued', min_value=0.0, step=1.0)
performance = st.number_input('Average Performance over the last 6 month', min_value=0.0, step=0.01)
localboqty = st.number_input('Current Unmet demand', min_value=0.0, step=1.0)
deck = st.radio('Is there a risk of mismatch between forecasted and actual demand of the product?', ['Yes','No'])
oe = st.radio('Did the product experience operational constraints?', ['Yes','No'])
ppap = st.radio('Is there a Production Part Approval Process risk associated with the product?', ['Yes','No'])
stop_auto = st.radio('Is automatic purchasing for this product halted?', ['Yes','No'])
rev = st.radio('Is review process for this product paused?', ['Yes','No'])

if st.button('Predict'):
    input_data = {
        "national_inv": inv,
        "lead_time": leadtime,
        "in_transit_qty": transit,
        "forecast_3_month": forecast,
        "sales_1_month": sales,
        "min_bank": minbank,
        "potential_issue": issue,
        "pieces_past_due": piecesdue,
        "perf_6_month_avg": performance,
        "local_bo_qty": localboqty,
        "deck_risk": deck,
        "oe_constraint": oe,
        "ppap_risk": ppap,
        "stop_auto_buy": stop_auto,
        "rev_stop": rev
    }

    # Send request to FastAPI endpoint
    API_URL = os.getenv("API_URL", "http://your-eb-url.com:8000/predict")
    response = requests.post(API_URL, json=input_data)
    
    # Get and display the prediction result
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Error in prediction!")

