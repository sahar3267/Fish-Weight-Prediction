def prep_data(df):

    X = df[["Height", "Width", "Length3", "Length2"]].values
    y = df["Weight"].values

    return X, y