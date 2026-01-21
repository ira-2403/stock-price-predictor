import pandas as pd
from sklearn.model_selection import train_test_split
def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("CSV file is empty")
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        if df.empty:
            raise ValueError("No valid numeric data after cleaning")
        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Close']
        return train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except Exception as e:
        print(f"❌ Error processing {csv_path}: {e}")
        return None, None, None, None
if __name__ == "__main__":
    for stock in ["AAPL", "GOOGL", "MSFT"]:
        print(f"\nProcessing {stock}")
        X_train, X_test, y_train, y_test = preprocess_data(f"data/{stock}_stock_data.csv")
        if X_train is not None:
            print("Train shape:", X_train.shape)
            print("Test shape:", X_test.shape)
        else:
            print(f"Skipping {stock}")
