from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np
def prep_data(df):

    df = df.assign(Girth=df["Length1"] * df["Width"])
    #df = df[["Species", "Length1", "Length2", "Length3", "Height", "Width", "Girth", "Weight" ]]

    X = df[["Length2", "Height", "Width", "Girth"]].values
    y = df["Weight"].values
    
    #ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    #X = np.array(ct.fit_transform(X))

    return X, y 