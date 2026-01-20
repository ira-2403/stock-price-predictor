from unittest import result
import joblib
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess_data import preprocess_data
def train_model():
    X_train,X_test,y_train,y_test = preprocess_data("data/AAPL_stock_data.csv")
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    print("Model Training Completed.")
    print("MAE: ",mae,"\n MSE: ",mse,"\n RMSE: ",rmse)
    result=pd.DataFrame({
    "Actual":y_test,
    "Predicted":y_pred
    })
    result.to_csv("result.csv",index=False)
    print("Saved predictions to results.csv")
    joblib.dump(model,"model.pkl")
    print("Model saved as model.pkl")
if __name__=="__main__":
    train_model()