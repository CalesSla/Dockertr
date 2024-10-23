import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

test_df = pd.read_csv("test.csv")
X_test, y_test = test_df["Feature"].values, test_df["Target"].values

model = joblib.load('linear_regression_model.joblib')

y_pred = model.predict(X_test.reshape(-1,1))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")