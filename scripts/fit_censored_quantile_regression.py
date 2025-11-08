import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from model.censored_quantile_regression_model import CensoredQuantileRegressionModel
from utils.data import load_feature_data
from utils.one_hot_feature_encoder import onehot_encode
from utils.draw import plot_summary_figure


def cqrm(endog, exog, tau=0.5, left=0, right=1000):
    X_const = sm.add_constant(exog)

    model = CensoredQuantileRegressionModel(
        endog, X_const, tau=tau, left=left, right=right
    )
    
    censored_quantile_regression_model_results = model.fit()

    plot_summary_figure(
        censored_quantile_regression_model_results.summary(),
        "outputs/figures/censored_quantile_regression_model_summary.png",
    )


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    feature_data_path = config["data"]["feature_data"]
    feature_data = load_feature_data(feature_data_path)

    feature_data = onehot_encode(feature_data)

    # Fit model
    y = feature_data["overall_score"]
    X = feature_data.drop(columns=["overall_score"])
    
    cqrm(endog=y, exog=X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()
    main(args.config)
