import pandas as pd 
from sklearn.model_selection import train_test_split
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df=df.dropna()
    X = df[['Open','High','Low','Volume']]
    y=df['Close']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test
if __name__=="__main__":
    X_train,X_test,y_train,y_test=preprocess_data("data/AAPL_stock_data.csv")
    print("Preprocessing completed.")
    print("Training data shape:",X_train.shape)
    print("Testing data shape:",X_test.shape)