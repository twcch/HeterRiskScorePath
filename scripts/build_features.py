import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import pandas as pd
from utils.data import load_processed_data, save_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV


def scaler_min_max(df):
    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = numeric_cols.drop("overall_score")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def feature_selection_based_on_lasso(df):
    y = df["overall_score"]
    X = df.drop(columns=["overall_score"])

    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)

    feature_importance = pd.DataFrame({"feature": X.columns, "coef": model.coef_})

    threshold = 0  # 排除 coef 為 0 的特徵

    feature_importance = feature_importance[
        abs(feature_importance["coef"]) <= threshold
    ]

    df = df.drop(columns=feature_importance.feature.values.tolist())

    return df


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    processed_data_path = config["data"]["processed_data"]
    processed_data = load_processed_data(processed_data_path)

    features_data = scaler_min_max(processed_data)

    features_data = feature_selection_based_on_lasso(features_data)

    save_data(features_data, config["data"]["feature_data"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()
    main(args.config)
