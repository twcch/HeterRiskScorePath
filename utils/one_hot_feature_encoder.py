import pandas as pd


def onehot_encode(df):
    non_numeric_cols = df.select_dtypes(exclude=["float64", "int64"]).columns
    df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

    return df
