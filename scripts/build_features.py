import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import pandas as pd
from datetime import datetime
from utils.data import load_processed_data, save_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        # 確保目錄存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # 清空或創建文件
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"特徵選擇日誌 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message):
        print(message)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

def scaler_min_max(df):
    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = numeric_cols.drop("overall_score")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def feature_selection_based_on_pearson(df, logger, threshold=0.8):
    """
    使用 Pearson 相關係數進行特徵選擇，排除高度相關的特徵
    
    Args:
        df: 包含 overall_score 的 DataFrame
        logger: Logger 對象
        threshold: 相關係數絕對值閾值，默認 0.8，高於此值的特徵將被排除
    
    Returns:
        特徵選擇後的 DataFrame
    """
    y = df["overall_score"]
    X = df.drop(columns=["overall_score"])

    logger.log(f"\n{'='*80}")
    logger.log(f"Pearson 相關係數特徵選擇（排除高相關性特徵）")
    logger.log(f"{'='*80}")
    logger.log(f"原始特徵數量: {X.shape[1]}")
    logger.log(f"原始特徵: {list(X.columns)}")
    logger.log(f"相關係數閾值: {threshold}\n")

    # 計算特徵之間的相關係數矩陣
    corr_matrix = X.corr().abs()
    
    # 找出高度相關的特徵對
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        logger.log(f"發現 {len(high_corr_pairs)} 對高度相關的特徵:")
        for pair in high_corr_pairs:
            logger.log(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
        logger.log("")
    else:
        logger.log("沒有發現相關係數超過閾值的特徵對\n")
    
    # 選擇要排除的特徵：對於每對高度相關的特徵，保留與 overall_score 相關性較高的
    features_to_drop = set()
    target_corr = X.corrwith(y).abs()
    
    for pair in high_corr_pairs:
        feat1, feat2 = pair['feature1'], pair['feature2']
        
        # 比較兩個特徵與目標變數的相關性，保留相關性較高的
        if target_corr[feat1] >= target_corr[feat2]:
            features_to_drop.add(feat2)
            logger.log(f"排除 {feat2} (與目標相關性: {target_corr[feat2]:.4f})，保留 {feat1} (與目標相關性: {target_corr[feat1]:.4f})")
        else:
            features_to_drop.add(feat1)
            logger.log(f"排除 {feat1} (與目標相關性: {target_corr[feat1]:.4f})，保留 {feat2} (與目標相關性: {target_corr[feat2]:.4f})")
    
    logger.log("")
    logger.log(f"被排除的特徵數量: {len(features_to_drop)}")
    logger.log(f"被排除的特徵: {list(features_to_drop)}\n")
    
    df = df.drop(columns=list(features_to_drop))
    
    logger.log(f"保留的特徵數量: {df.shape[1] - 1}")  # -1 因為包含 overall_score
    logger.log(f"保留的特徵: {list(df.drop(columns=['overall_score']).columns)}")
    logger.log(f"{'='*80}\n")

    return df

def feature_selection_based_on_lasso(df, logger):
    y = df["overall_score"]
    X = df.drop(columns=["overall_score"])

    logger.log(f"\n{'='*80}")
    logger.log(f"LASSO 特徵選擇")
    logger.log(f"{'='*80}")
    logger.log(f"原始特徵數量: {X.shape[1]}")
    logger.log(f"原始特徵: {list(X.columns)}\n")

    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)

    feature_importance = pd.DataFrame({"feature": X.columns, "coef": model.coef_})

    threshold = 0  # 排除 coef 為 0 的特徵

    excluded_features = feature_importance[
        abs(feature_importance["coef"]) <= threshold
    ]
    
    logger.log(f"被排除的特徵數量: {len(excluded_features)}")
    logger.log(f"被排除的特徵: {list(excluded_features['feature'].values)}\n")
    
    df = df.drop(columns=excluded_features.feature.values.tolist())
    
    logger.log(f"保留的特徵數量: {df.shape[1] - 1}")  # -1 因為包含 overall_score
    logger.log(f"保留的特徵: {list(df.drop(columns=['overall_score']).columns)}")
    logger.log(f"{'='*80}\n")

    return df

def feature_selection_based_on_pca(df, logger, variance_threshold=0.95):
    """
    使用 PCA 進行特徵降維
    
    Args:
        df: 包含 overall_score 的 DataFrame
        logger: Logger 對象
        variance_threshold: 保留的累積方差比例，默認 0.95
    
    Returns:
        降維後的 DataFrame，包含主成分和 overall_score
    """
    y = df["overall_score"]
    X = df.drop(columns=["overall_score"])
    
    logger.log(f"\n{'='*80}")
    logger.log(f"PCA 特徵降維")
    logger.log(f"{'='*80}")
    logger.log(f"原始特徵數量: {X.shape[1]}")
    logger.log(f"原始特徵: {list(X.columns)}")
    logger.log(f"保留方差比例閾值: {variance_threshold}\n")
    
    # 使用 PCA 進行降維
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X)
    
    logger.log(f"降維後主成分數量: {X_pca.shape[1]}")
    logger.log(f"各主成分解釋的方差比例:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_, 1):
        logger.log(f"  PC{i}: {var_ratio:.4f}")
    logger.log(f"\n累積解釋方差比例: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 創建新的 DataFrame
    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
    df_pca["overall_score"] = y.values
    
    logger.log(f"\n所有原始特徵已轉換為 {len(pca_columns)} 個主成分")
    logger.log(f"主成分名稱: {pca_columns}")
    logger.log(f"{'='*80}\n")
    
    return df_pca


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 創建 logger
    log_path = config.get("logs", {}).get("feature_selection_log", "outputs/logs/feature_selection.txt")
    logger = Logger(log_path)
    
    processed_data_path = config["data"]["processed_data"]
    processed_data = load_processed_data(processed_data_path)

    features_data = scaler_min_max(processed_data)
    
    features_data = feature_selection_based_on_pearson(features_data, logger)
    
    if config["experiment"]["feature_selection_method"] == "pca":
        features_data = feature_selection_based_on_pca(features_data, logger)
        save_data(features_data, config["data"]["feature_data_pca"])
        logger.log(f"PCA 特徵數據已保存至: {config['data']['feature_data_pca']}")
    else:
        features_data = feature_selection_based_on_lasso(features_data, logger)
        save_data(features_data, config["data"]["feature_data_lasso"])
        logger.log(f"LASSO 特徵數據已保存至: {config['data']['feature_data_lasso']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()
    main(args.config)
