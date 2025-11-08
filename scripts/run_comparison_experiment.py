import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# 將專案根目錄加入路徑以匯入自定義模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.censored_quantile_regression_model import CensoredQuantileRegressionModel
# 假設你有這個 Tobit 模型 (如果沒有，可以用 statsmodels 的替代)
from model.censored_regression_model import CensoredRegressionModel
from utils.data import load_feature_data
from utils.one_hot_feature_encoder import onehot_encode

def perform_lasso_selection(X, y, random_state=42):
    """
    使用 LassoCV 自動選擇最佳特徵子集
    """
    print("正在執行 LASSO 特徵選擇...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用交叉驗證自動選擇最佳正則化參數 alpha
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # 選出係數不為 0 的特徵
    selected_mask = lasso.coef_ != 0
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"LASSO 選擇了 {len(selected_features)}/{X.shape[1]} 個特徵。")
    return selected_features

def extract_model_results(res, model_prefix, base_feature_names):
    """
    穩健地從結果物件中提取係數。自動處理可能有額外 sigma 參數的情況。
    """
    # 確保我們拿到的是純 numpy array，避免舊索引干擾
    params = res.params.values if hasattr(res.params, 'values') else np.array(res.params)
    bse = res.bse.values if hasattr(res.bse, 'values') else np.array(res.bse)
    pvalues = res.pvalues.values if hasattr(res.pvalues, 'values') else np.array(res.pvalues)

    # 自動偵測並調整索引名稱
    current_names = list(base_feature_names)
    if len(params) == len(base_feature_names) + 1:
        # 如果參數多一個，通常是 sigma (尺度參數)
        current_names.append("sigma (scale)")
    elif len(params) != len(base_feature_names):
        print(f"警告: {model_prefix} 參數數量 ({len(params)}) 與特徵數量 ({len(base_feature_names)}) 不匹配，將使用數字索引。")
        current_names = None 

    # 建立該模型的結果 DataFrame
    df = pd.DataFrame({
        f'{model_prefix}_coef': params,
        f'{model_prefix}_se': bse,
        f'{model_prefix}_pval': pvalues
    }, index=current_names)
    
    return df

def main(config_path):
    # 1. 載入設定與資料
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 根據設定決定從哪裡讀取資料
    if config["experiment"].get("feature_selection_method") == "lasso":
        # 如果設定檔指定了 lasso，優先嘗試讀取已經做過初步篩選的資料
        data_path = config["data"].get("feature_data_lasso", config["data"]["processed_data"])
    else:
        data_path = config["data"]["processed_data"]
        
    print(f"正在讀取資料: {data_path}")
    df = load_feature_data(data_path)
    df = onehot_encode(df)

    y = df["overall_score"]
    X_raw = df.drop(columns=["overall_score"])

    # 2. LASSO 特徵選擇
    # 為了確保實驗的嚴謹性，這裡再執行一次 LASSO 來鎖定最終要比較的特徵集
    selected_features = perform_lasso_selection(X_raw, y)
    X_selected = X_raw[selected_features]
    
    # 加入常數項 (截距)
    X_const = sm.add_constant(X_selected)
    base_feature_names = X_const.columns.tolist()
    
    # 初始化主比較表 (使用 join='outer' 可以自動處理 sigma 行的有無)
    final_comparison = pd.DataFrame(index=base_feature_names)

    # ============================================
    # 模型 1: Multiple Linear Regression (OLS)
    # ============================================
    print("\n--- Fitting Multiple Linear Regression (OLS) ---")
    ols_model = sm.OLS(y, X_const)
    ols_res = ols_model.fit()
    final_comparison = final_comparison.join(
        extract_model_results(ols_res, "OLS", base_feature_names), 
        how='outer'
    )

    # ============================================
    # 模型 2: Censored Regression (Tobit)
    # ============================================
    print("\n--- Fitting Censored Regression (Tobit) ---")
    try:
        tobit_model = CensoredRegressionModel(y, X_const, left=0, right=1000)
        tobit_res = tobit_model.fit(disp=0)
        final_comparison = final_comparison.join(
            extract_model_results(tobit_res, "Tobit", base_feature_names), 
            how='outer'
        )
    except Exception as e:
        print(f"Tobit 模型擬合失敗: {e}")

    # ============================================
    # 模型 3: Censored Quantile Regression (CQR)
    # ============================================
    # 從設定檔讀取 tau 列表，如果沒有就用預設值
    taus = config["experiment"].get("censored_quantile_regression", {}).get("tau_list", [0.1, 0.5, 0.9])
    
    for t in taus:
        print(f"\n--- Fitting CQR (tau={t}) ---")
        try:
            cqr_model = CensoredQuantileRegressionModel(y, X_const, tau=t, left=0, right=1000)
            
            print(f"執行 Bootstrap (n=100) 估計 tau={t} 的標準誤...")
            # 使用之前修好的 fit_bootstrap
            cqr_res = cqr_model.fit_bootstrap(n_boot=100) 
            
            prefix = f"CQR_{int(t*100)}"
            final_comparison = final_comparison.join(
                extract_model_results(cqr_res, prefix, base_feature_names), 
                how='outer'
            )
        except Exception as e:
            print(f"CQR (tau={t}) 模型擬合失敗: {e}")

    # ============================================
    # 4. 匯出結果
    # ============================================
    output_dir = "outputs/tables"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_comparison_lasso.csv")
    
    final_comparison.index.name = "Feature"
    # 將 sigma (scale) 移動到最後一列 (美觀考量，可選)
    if "sigma (scale)" in final_comparison.index:
        new_index = [i for i in final_comparison.index if i != "sigma (scale)"] + ["sigma (scale)"]
        final_comparison = final_comparison.reindex(new_index)

    final_comparison.to_csv(output_path)
    
    print(f"\n所有模型擬合完畢！比較表格已匯出至: {output_path}")
    # print("預覽結果:")
    # print(final_comparison)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行完整模型比較實驗")
    parser.add_argument("--config", default="configs/config.json", help="設定檔路徑")
    args = parser.parse_args()
    
    main(args.config)