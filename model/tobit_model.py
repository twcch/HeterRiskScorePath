import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm


class TobitModel(GenericLikelihoodModel):
    def __init__(self, endog, exog, left=0, right=1000, **kwargs):
        super(TobitModel, self).__init__(endog, exog, **kwargs)
        self.left = left
        self.right = right

    # 定義負對數概似函數 (Negative Log-Likelihood)
    def nloglikeobs(self, params):
        exog = self.exog
        endog = self.endog
        beta = params[:-1]  # 迴歸係數
        sigma = params[-1]  # 標準差

        # 確保 sigma 為正值
        sigma = np.abs(sigma) + 1e-10

        mu = np.dot(exog, beta)

        # 標記數據狀態
        left_mask = endog <= self.left
        right_mask = endog >= self.right
        uncensored_mask = ~(left_mask | right_mask)

        ll = np.zeros_like(endog)

        # 1. 未刪失數據的 Log-Likelihood (類似 OLS)
        if np.any(uncensored_mask):
            ll[uncensored_mask] = norm.logpdf(
                (endog[uncensored_mask] - mu[uncensored_mask]) / sigma
            ) - np.log(sigma)

        # 2. 左刪失數據的 Log-Likelihood (使用 CDF)
        if np.any(left_mask):
            ll[left_mask] = norm.logcdf((self.left - mu[left_mask]) / sigma)

        # 3. 右刪失數據的 Log-Likelihood (使用 1-CDF，即 Survival Function)
        if np.any(right_mask):
            ll[right_mask] = norm.logsf((self.right - mu[right_mask]) / sigma)

        return -ll  # 返回負值供最小化使用

    # 模型擬合
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # 使用 OLS 結果作為初始參數猜測值，加速收斂
        if start_params is None:
            ols = sm.OLS(self.endog, self.exog).fit()
            start_params = np.append(ols.params, ols.resid.std())

        return super(TobitModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )
