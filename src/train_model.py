import joblib
import numpy as np
import os
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess_data import preprocess_data
def train_model(stock):
    csv_path=f"data/{stock}_stock_data.csv"
    X_train,X_test,y_train,y_test = preprocess_data(csv_path)
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    print("Model Training Completed.")
    print("MAE: ",mae,"\n MSE: ",mse,"\n RMSE: ",rmse)
    result=pd.DataFrame({
    "Stock":stock,
    "Date":y_test.index,
    "Actual":y_test.values,
    "Predicted":y_pred
    })
    file_exists=os.path.exists("result.csv")
    result.to_csv(
        "result.csv",
        index=False,
        mode="a" if file_exists else "w",
        header=not file_exists
        )
    print("Saved predictions to result.csv")
    joblib.dump(model,f"model_{stock}.pkl")
    print(f"Model saved as model_{stock}.pkl")
if __name__=="__main__":
    train_model("AAPL")
    train_model("GOOGL")
    train_model("MSFT")