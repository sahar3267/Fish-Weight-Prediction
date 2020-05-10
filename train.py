import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump
from preprocess import prep_data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np


   
df = pd.read_csv("fish_participant.csv")
X, y = prep_data(df)

regressor = LinearRegression()
regressor.fit(X, y)

dump(regressor, "reg.joblib")