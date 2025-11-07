import pandas as pd
from pathlib import Path

raw_data_types = {
    "姓名": "string",
    "年齡": "float64",
    "性別": "category",
    "好信一鑒通": "category",
    "手機入網": "category",
    "手機實名制": "category",
    "反詐欺風險": "category",
    "身份特徵": "float64",
    "履約能力": "float64",
    "失信風險": "float64",
    "消費偏好": "float64",
    "行為特徵": "float64",
    "社交影響": "float64",
    "成長潛力": "float64",
    "機構所屬行業": "category",
    "命中機構數": "float64",
    "命中銀行數": "float64",
    "命中消費金融數": "float64",
    "命中小貸機構數": "float64",
    "機構查詢總次數": "float64",
    "近三個月機構查詢次數": "float64",
    "近六個月機構查詢次數": "float64",
    "好信風險": "float64",
    "好信涉訴": "float64",
    "綜合評分": "float64",
    "業務時間": "string",
}


def load_raw_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=raw_data_types)

    return df


def load_processed_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=raw_data_types)

    return df


def load_processed_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=raw_data_types)

    return df


def load_feature_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=raw_data_types)

    return df


def save_data(df: pd.DataFrame, file_path: Path):
    df.to_csv(file_path, index=False)
