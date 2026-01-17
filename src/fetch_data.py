import yfinance as yf
import pandas as pd
def fetch_stock_data(ticker,start_date,end_date):
    stock=yf.download(ticker,start=start_date,end=end_date)
    return stock
if __name__=="__main__":
    ticker="AAPL"
    start_date="2018-01-01"
    end_date="2025-12-31"
    df=fetch_stock_data(ticker,start_date,end_date)
    if df.empty:
        print("No data found. Check ticker symbol.")
    else:
        df.to_csv(f"data/{ticker}_stock_data.csv")
        print(f"Data saved to data/{ticker}_stock_data.csv")