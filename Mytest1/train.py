import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

train_df = pd.read_csv("train.csv")

X_train, y_train = train_df["Feature"].values,  train_df["Target"].values

model = LinearRegression()
model.fit(X_train.reshape(-1,1), y_train)

joblib.dump(model, 'linear_regression_model.joblib')