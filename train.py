import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from preprocess import prep_data
from joblib import dump
import os

# Import Training data
path = os.path.join('fish_participant.csv')
df = pd.read_csv(path)

# create model - we will use a gradient boosting regressor
X, y = prep_data(df)

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=0, loss='ls').fit(X, y)

dump(est, "est.joblib")