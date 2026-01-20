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
- data/
- src/
- app.py
- fetch_data.py
- preprocess_data.py
- train_model.py
- requirements.txt
- result.csv

## Features
- Stock price prediction
- Model evaluation (MAE,RMSE)
- Interactive Streamlit web app
- Visualization of predictions

## How to Run
'''bash
pip install -r requirements.txt
python src/fetch_data.py
python src/train_model.py
streamlit run src/app.py
'''

## Future Improvement
- LSTM model
- Multi-stock comparison
- Live deployment