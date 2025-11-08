import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from model.multiple_linear_regression_model import MultipleLinearRegressionModel
from utils.data import load_feature_data
from utils.one_hot_feature_encoder import onehot_encode
from utils.draw import plot_summary_figure


def mlrm(feature_selection_method, y, X):
    X_const = sm.add_constant(X)

    multiple_linear_regression_model = MultipleLinearRegressionModel(y, X_const)
    multiple_linear_regression_model_results = multiple_linear_regression_model.fit()

    plot_summary_figure(
        multiple_linear_regression_model_results.summary(), f"outputs/figures/{feature_selection_method}/mlrm_summary.png"
    )


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    feature_selection_method = config["experiment"]["feature_selection_method"]

    if feature_selection_method == "lasso":
        feature_data_path = config["data"]["feature_data_lasso"]
    elif feature_selection_method == "pca":
        feature_data_path = config["data"]["feature_data_pca"]

    feature_data = load_feature_data(feature_data_path)

    feature_data = onehot_encode(feature_data)
    
    # Fit model
    y = feature_data["overall_score"]
    X = feature_data.drop(columns=["overall_score"])
    mlrm(feature_selection_method, y, X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()
    main(args.config)
