import streamlit as st 
import numpy as np
import joblib
import pandas as pd 
import matplotlib.pyplot as plt
st.set_page_config(page_title="Stock Price Predictor",layout="centered")
stock=st.selectbox("Select Stock",["AAPL","GOOGL","MSFT"])
model=joblib.load(f"model_{stock}.pkl")
st.title("Stock Price Predictor")
st.write("Predict **closing price** using Random Forest Model")
st.sidebar.header("Input Stock Values")
open_price=st.sidebar.number_input("Open Price",min_value=0.0,value=100.0)
high_price=st.sidebar.number_input("High Price",min_value=0.0,value=105.0)
low_price=st.sidebar.number_input("Low Price",min_value=0.0,value=95.0)
volume=st.sidebar.number_input("Volume",min_value=0)
st.subheader("Model Performance")
try:
    df=pd.read_csv("result.csv")
    df=df[df["Stock"]==stock]
    if df.empty:
        st.warning("No data available for this stock")
    else:
        fig,ax=plt.subplots()
        ax.scatter(df["Actual"],df["Predicted"],color="green",alpha=0.5)
        ax.set_xlabel("Actual Close Price")
        ax.set_ylabel("Predicted Close Price")
        ax.set_title(f"{stock} - Actual vs Predicted Close Price")
        min_val=min(df["Actual"].min(),df["Predicted"].min())
        max_val=max(df["Actual"].max(),df["Predicted"].max())
        ax.plot([min_val,max_val],[min_val,max_val], "r-")
        st.pyplot(fig)
except Exception as e:
    st.info("Run train_model.py to generate results.csv")
if st.sidebar.button("Predict Closing Price"):
    input_data=np.array([[open_price,high_price,low_price,volume]])
    prediction=model.predict(input_data)
    st.success(f"Predicted Close Price: ${prediction[0]:.2f}")