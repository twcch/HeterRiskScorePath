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


def cqrm(feature_selection_method, endog, exog, tau=0.5, left=0, right=1000):
    # 確保 exog 是 DataFrame
    if not isinstance(exog, pd.DataFrame):
         exog = pd.DataFrame(exog)
         
    X_const = sm.add_constant(exog)
    # 再次確認 X_const 仍是 DataFrame 且有 columns
    # print(X_const.columns) 
    
    model = CensoredQuantileRegressionModel(
        endog, X_const, tau=tau, left=left, right=right
    )
    
    print("正在使用 Bootstrap 估計真實的 P 值，這需要一點時間...")
    # 強制使用 Bootstrap，跑 100 次 (正式報告建議 200-500 次)
    # 這會取代原本那個快樂表結果
    censored_quantile_regression_model_results = model.fit_bootstrap(n_boot=100)

    plot_summary_figure(
        censored_quantile_regression_model_results.summary(),
        f"outputs/figures/{feature_selection_method}/cqrm_tau{tau}_summary.png",
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

    for tau in config["experiment"]["censored_quantile_regression"]["tau_list"]:
        cqrm(feature_selection_method, endog=y, exog=X, tau=tau)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()
    main(args.config)
