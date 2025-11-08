import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm
from scipy.optimize import minimize


class TruncatedQuantileRegressionModel(GenericLikelihoodModel):
    """
    截斷分位數迴歸模型 (Truncated Quantile Regression)
    
    Parameters:
    -----------
    endog : array-like
        因變數 (y)
    exog : array-like
        自變數矩陣 (X)
    tau : float
        分位數水平，範圍 (0, 1)，例如 0.5 表示中位數
    left : float
        左截斷點，預設為 0
    right : float
        右截斷點，預設為 1000
    """
    def __init__(self, endog, exog, tau=0.5, left=0, right=1000, **kwargs):
        super(TruncatedQuantileRegressionModel, self).__init__(endog, exog, **kwargs)
        self.tau = tau  # 分位數水平
        self.left = left
        self.right = right
        
        # 驗證分位數參數
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1")
    
    def check_loss(self, u, tau):
        """
        分位數損失函數 (Check Loss / Pinball Loss)
        ρ_τ(u) = u * (τ - I(u < 0))
        """
        return u * (tau - (u < 0).astype(float))
    
    def nloglikeobs(self, params):
        """
        負對數擬似概似函數
        使用 Powell (1986) 的截斷分位數迴歸方法
        """
        exog = self.exog
        endog = self.endog
        beta = params[:-1]  # 迴歸係數
        sigma = np.abs(params[-1]) + 1e-10  # 標準差（尺度參數）
        
        mu = np.dot(exog, beta)
        
        # 標記數據狀態
        left_mask = endog <= self.left
        right_mask = endog >= self.right
        uncensored_mask = ~(left_mask | right_mask)
        
        ll = np.zeros_like(endog)
        
        # 1. 未截斷數據：使用分位數損失函數
        if np.any(uncensored_mask):
            residuals = endog[uncensored_mask] - mu[uncensored_mask]
            # 使用非對稱 Laplace 分佈的對數概似
            ll[uncensored_mask] = self._asymmetric_laplace_logpdf(
                residuals, self.tau, sigma
            )
        
        # 2. 左截斷數據：使用累積分佈函數
        if np.any(left_mask):
            z_left = (self.left - mu[left_mask]) / sigma
            ll[left_mask] = self._quantile_cdf_log(z_left, self.tau)
        
        # 3. 右截斷數據：使用生存函數 (1 - CDF)
        if np.any(right_mask):
            z_right = (self.right - mu[right_mask]) / sigma
            ll[right_mask] = self._quantile_sf_log(z_right, self.tau)
        
        return -ll  # 返回負值供最小化使用
    
    def _asymmetric_laplace_logpdf(self, u, tau, sigma):
        """
        非對稱 Laplace 分佈的對數概似函數
        用於分位數迴歸的概似推論
        """
        theta = (1 - 2 * tau) / (tau * (1 - tau))
        kappa = 2 / (tau * (1 - tau))
        
        logpdf = np.log(tau * (1 - tau) / sigma) - (kappa / sigma) * self.check_loss(u, tau)
        return logpdf
    
    def _quantile_cdf_log(self, z, tau):
        """
        分位數模型的累積分佈函數對數值
        使用非對稱 Laplace 分佈近似
        """
        # 簡化處理：使用標準常態分佈的 CDF
        return norm.logcdf(z)
    
    def _quantile_sf_log(self, z, tau):
        """
        分位數模型的生存函數對數值
        """
        return norm.logsf(z)
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method='bfgs', **kwargs):
        """
        模型擬合
        
        Parameters:
        -----------
        start_params : array-like, optional
            初始參數值
        maxiter : int
            最大迭代次數
        maxfun : int
            最大函數評估次數
        method : str
            優化方法，建議使用 'bfgs' 或 'powell'
        """
        # 使用分位數迴歸結果作為初始參數
        if start_params is None:
            try:
                # 使用 statsmodels 的分位數迴歸作為初始值
                from statsmodels.regression.quantile_regression import QuantReg
                qr = QuantReg(self.endog, self.exog).fit(q=self.tau)
                residuals = self.endog - np.dot(self.exog, qr.params)
                sigma_init = np.percentile(np.abs(residuals), 75) / 0.6745  # MAD estimator
                start_params = np.append(qr.params, sigma_init)
            except:
                # 備用方案：使用 OLS
                ols = sm.OLS(self.endog, self.exog).fit()
                start_params = np.append(ols.params, ols.resid.std())
        
        return super(TruncatedQuantileRegressionModel, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            method=method,
            **kwargs
        )
    
    def predict_quantile(self, exog=None, quantile=None):
        """
        預測指定分位數
        
        Parameters:
        -----------
        exog : array-like, optional
            自變數矩陣，如果為 None 則使用訓練數據
        quantile : float, optional
            要預測的分位數，如果為 None 則使用模型的 tau
        
        Returns:
        --------
        predictions : array
            預測的分位數值
        """
        if exog is None:
            exog = self.exog
        
        if quantile is None:
            quantile = self.tau
        
        beta = self.params[:-1]
        predictions = np.dot(exog, beta)
        
        # 確保預測值在截斷範圍內
        predictions = np.clip(predictions, self.left, self.right)
        
        return predictions


# # 使用範例
# if __name__ == "__main__":
#     # 生成模擬數據
#     np.random.seed(42)
#     n = 500
#     X = np.random.randn(n, 3)
#     X = sm.add_constant(X)
    
#     # 真實參數
#     true_beta = np.array([5, 1.5, -0.8, 2.0])
    
#     # 生成異質性誤差
#     epsilon = np.random.standard_t(df=5, size=n)
#     y_latent = np.dot(X, true_beta) + epsilon
    
#     # 應用截斷
#     left_truncation = 3
#     right_truncation = 10
#     y = np.clip(y_latent, left_truncation, right_truncation)
    
#     # 擬合不同分位數的模型
#     quantiles = [0.25, 0.5, 0.75]
#     models = {}
    
#     for tau in quantiles:
#         print(f"\n擬合分位數 τ = {tau}")
#         model = TruncatedQuantileRegressionModel(
#             y, X, tau=tau, 
#             left=left_truncation, 
#             right=right_truncation
#         )
#         result = models[tau] = model.fit()
#         print(result.summary())
#         print(f"估計係數: {result.params[:-1]}")
#         print(f"估計標準差: {result.params[-1]}")