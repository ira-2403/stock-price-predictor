import streamlit as st 
import numpy as np
import joblib
model=joblib.load("model.pkl")
st.set_page_config(page_title="Stock Price Predictor",layout="centered")
st.title("Stock Price Predictor")
st.write("Predict **closing price** using Linear Regression Model")
st.sidebar.header("Input Stock Values")
open_price=st.sidebar.number_input("Open Price",min_value=0.0,value=100.0)
high_price=st.sidebar.number_input("High Price",min_value=0.0,value=105.0)
low_price=st.sidebar.number_input("Low Price",min_value=0.0,value=95.0)
volume=st.sidebar.number_input("Volume",min_value=0)
if st.sidebar.button("Predict Closing Price"):
    input_data=np.array([[open_price,high_price,low_price,volume]])
    prediction=model.predict(input_data)
    st.success(f"Predicted Close Price: ${prediction[0]:.2f}")
