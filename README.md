# 📈 Stock Price Predictor

A machine learning based project that predicts the closing price of a stock using historical market data using Linear Regression and Streamlit.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- yFinance
- Streamlit

## Project Structure
- .devcontainer
- .venv
- .vscode
- data/
- AAPL_stock_data.csv
- GOOGL_stock_data.csv
- MSFT_stock_data.csv
- src/
- app.py
- fetch_data.py
- preprocess_data.py
- train_model.py
- model_AAPL.pkl
- model_GOOGL.pkl
- model_MSFT.pkl
- requirements.txt
- result.csv
- runtime.txt

## Unique Selling Point
- End-to-end ML pipeline: data fetching->preprocessing->training->delopment
- Multi-stock support using separate trained models
- Live, interactive UI for predictions
- Clear model preformance visualization (Actual vs Predicted)
- Beginner-friendly yet production-style project structure
- Deployed and publicly accessible via Streamlit Cloud

## Features
- Stock price prediction
- Model evaluation (MAE,RMSE)
- Interactive Streamlit web app
- Visualization of predictions

## Supported Stocks
- Apple (APPL)
- Google (GOOGL)
- Microsoft (MSFT)

## How to Run
- pip install -r requirements.txt
- python src/fetch_data.py
- python src/train_model.py
- streamlit run src/app.py

## Deployed
- https://stock-price-predictor-9vyjy3njv37n2upgdceasj.streamlit.app/

## Future Improvement
- LSTM model
