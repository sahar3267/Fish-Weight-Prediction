### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
import pandas as pd

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    X, y = prep_data(df)

    reg = load("reg.joblib")

    predictions = reg.predict(X)

    return predictions

if __name__ == "__main__":
    predictions = predict_from_csv("fish_holdout_demo.csv")
    print(predictions)
######

### WE WRITE THIS ###
    #from sklearn.metrics import mean_squared_error
    #ho_predictions = predict_from_csv("fish_holdout.csv")
    #ho_truth = pd.read_csv("fish_holdout.csv")["Weight"].values
   # ho_mse = mean_squared_error(ho_truth, ho_predictions)
    #print(ho_mse)
###########

