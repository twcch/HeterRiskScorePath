import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse

import pandas as pd
from utils.data import load_raw_data, save_data


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_mapping = {
        "姓名": "name",
        "年齡": "age",
        "性別": "gender",
        "好信一鑒通": "haoxin_yijiantong",
        "手機入網": "mobile_network_registered",
        "手機實名制": "mobile_realname_verification",
        "反詐欺風險": "fraud_risk",
        "身份特徵": "identity_feature_score",
        "履約能力": "contract_ability_score",
        "失信風險": "default_risk_score",
        "消費偏好": "consumer_preference_score",
        "行為特徵": "behavior_feature_score",
        "社交影響": "social_influence_score",
        "成長潛力": "growth_potential_score",
        "機構所屬行業": "institution_industry",
        "命中機構數": "hit_institutions_count",
        "命中銀行數": "hit_banks_count",
        "命中消費金融數": "hit_consumer_finance_count",
        "命中小貸機構數": "hit_microloan_count",
        "機構查詢總次數": "institution_query_total",
        "近三個月機構查詢次數": "institution_query_3m",
        "近六個月機構查詢次數": "institution_query_6m",
        "好信風險": "haoxin_risk_score",
        "好信涉訴": "haoxin_litigation_score",
        "綜合評分": "overall_score",
        "業務時間": "business_time",
    }

    df = df.rename(columns=columns_mapping)

    return df


def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.drop(columns=columns)

    return df


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()

    return df


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    raw_data_path = config["data"]["raw_data"]
    raw_data = load_raw_data(raw_data_path)

    cleaned_data = rename_columns(raw_data)
    cleaned_data = remove_columns(cleaned_data, ["name", "business_time"])
    cleaned_data = remove_missing_values(cleaned_data)

    processed_data_path = config["data"]["processed_data"]
    save_data(cleaned_data, processed_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()
    main(args.config)
