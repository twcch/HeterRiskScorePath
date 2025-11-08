import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm
from scipy import stats # 確保 scipy.stats 被導入

class CensoredQuantileRegressionModel(GenericLikelihoodModel):
    """
    設限分位數迴歸模型 (Censored Quantile Regression Model, CQR)
    基於非對稱拉普拉斯分佈 (ALD) 概似函數。
    """
    def __init__(self, endog, exog, tau=0.5, left=-np.inf, right=np.inf, **kwargs):
        self.original_endog = endog
        self.original_exog = exog
        
        self.exog_names_custom = None
        if isinstance(exog, pd.DataFrame):
            self.exog_names_custom = exog.columns.tolist()
        elif isinstance(exog, pd.Series):
             self.exog_names_custom = [exog.name]

        endog_np = np.asarray(endog).flatten()
        exog_np = np.asarray(exog)
        
        super(CensoredQuantileRegressionModel, self).__init__(endog_np, exog_np, **kwargs)
        
        if self.exog_names_custom:
             if self.k_constant == 1 and len(self.exog_names_custom) == self.exog.shape[1] - 1:
                  self.data.xnames = ['const'] + self.exog_names_custom
             elif len(self.exog_names_custom) == self.exog.shape[1]:
                  self.data.xnames = self.exog_names_custom

        self.tau = tau
        self.left = left
        self.right = right
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1")

    def check_loss(self, u):
        return u * (self.tau - (u < 0).astype(float))

    def _ald_logpdf(self, u, sigma):
        sigma = np.maximum(sigma, 1e-10)
        const_term = np.log(self.tau * (1 - self.tau)) - np.log(sigma)
        exp_term = -self.check_loss(u) / sigma
        return const_term + exp_term

    def _ald_logcdf(self, u, sigma):
        sigma = np.maximum(sigma, 1e-10)
        z = u / sigma
        logcdf = np.zeros_like(u)
        mask_neg = u < 0
        mask_pos = ~mask_neg
        if np.any(mask_neg):
            logcdf[mask_neg] = np.log(self.tau) + (1 - self.tau) * z[mask_neg]
        if np.any(mask_pos):
            val = (1 - self.tau) * np.exp(-self.tau * z[mask_pos])
            val = np.minimum(val, 1.0 - 1e-10)
            logcdf[mask_pos] = np.log1p(-val)
        return logcdf

    def _ald_logsf(self, u, sigma):
        sigma = np.maximum(sigma, 1e-10)
        z = u / sigma
        logsf = np.zeros_like(u)
        mask_neg = u < 0
        mask_pos = ~mask_neg
        if np.any(mask_pos):
            logsf[mask_pos] = np.log(1 - self.tau) - self.tau * z[mask_pos]
        if np.any(mask_neg):
            val = self.tau * np.exp((1 - self.tau) * z[mask_neg])
            val = np.minimum(val, 1.0 - 1e-10)
            logsf[mask_neg] = np.log1p(-val)
        return logsf

    def nloglikeobs(self, params):
        beta = params[:-1]
        sigma = np.exp(params[-1])
        mu = np.dot(self.exog, beta)
        resid = self.endog - mu
        left_mask = (self.endog <= self.left)
        right_mask = (self.endog >= self.right)
        uncensored_mask = ~(left_mask | right_mask)
        ll = np.zeros_like(self.endog)
        if np.any(uncensored_mask):
            ll[uncensored_mask] = self._ald_logpdf(resid[uncensored_mask], sigma)
        if np.any(left_mask):
            ll[left_mask] = self._ald_logcdf(self.left - mu[left_mask], sigma)
        if np.any(right_mask):
             ll[right_mask] = self._ald_logsf(self.right - mu[right_mask], sigma)
        return -ll

    def fit(self, start_params=None, maxiter=5000, maxfun=5000, method='bfgs', **kwargs):
        if start_params is None:
            try:
                qr_res = sm.QuantReg(self.endog, self.exog).fit(q=self.tau)
                resid = self.endog - qr_res.predict(self.exog)
                sigma_init = np.median(np.abs(resid - np.median(resid))) / 0.6745
                start_params = np.append(qr_res.params, np.log(np.maximum(sigma_init, 0.1)))
            except:
                ols_res = sm.OLS(self.endog, self.exog).fit()
                start_params = np.append(ols_res.params, np.log(np.std(ols_res.resid)))

        # 為了避免在 Bootstrap 過程中印出太多訊息，我們檢查 'disp' 參數
        disp = kwargs.get('disp', 1)
        if disp:
             print(f"Fitting CQR (tau={self.tau}) with {method}...")
             
        try:
            res = super(CensoredQuantileRegressionModel, self).fit(
                start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=method, **kwargs
            )
            if np.any(np.isnan(res.bse)) and method == 'bfgs':
                if disp: print("BFGS produced NaNs, retrying with Nelder-Mead...")
                res = super(CensoredQuantileRegressionModel, self).fit(
                    start_params=res.params, maxiter=maxiter, maxfun=maxfun, method='nm', **kwargs
                )
            return res
        except Exception as e:
            if disp: print(f"Optimization failed: {e}")
            return None

    def fit_bootstrap(self, n_boot=100, start_params=None, **kwargs):
        """
        使用 Bootstrap 方法估計標準誤。
        """
        print(f"Starting Bootstrap estimation with {n_boot} iterations...")
        
        res_point = self.fit(start_params=start_params, disp=0, **kwargs)
        if res_point is None:
             raise RuntimeError("Initial point estimation failed.")
             
        point_params = res_point.params
        n = len(self.endog)
        boot_params = []

        for i in range(n_boot):
            if (i + 1) % 10 == 0: print(f"Bootstrap iteration {i+1}/{n_boot}")
            indices = np.random.choice(n, n, replace=True)
            
            if isinstance(self.original_endog, (pd.Series, pd.DataFrame)):
                 y_boot = self.original_endog.iloc[indices]
            else:
                 y_boot = self.endog[indices]
                 
            if isinstance(self.original_exog, (pd.Series, pd.DataFrame)):
                 X_boot = self.original_exog.iloc[indices]
            else:
                 X_boot = self.exog[indices]

            try:
                boot_model = CensoredQuantileRegressionModel(
                    y_boot, X_boot, tau=self.tau, left=self.left, right=self.right
                )
                res_boot = boot_model.fit(start_params=point_params, maxiter=1000, disp=0, method='bfgs')
                
                if res_boot is not None and not np.any(np.isnan(res_boot.params)):
                    boot_params.append(res_boot.params)
            except:
                continue

        if len(boot_params) < n_boot * 0.5:
             print("Warning: More than 50% of bootstrap iterations failed.")

        boot_params = np.array(boot_params)
        # 處理可能出現的奇異協方差矩陣，加一點點噪音到對角線
        try:
            boot_cov = np.cov(boot_params.T)
            boot_bse = np.sqrt(np.diag(boot_cov))
        except:
             # 如果還是失敗，用簡單的標準差代替
             boot_bse = np.std(boot_params, axis=0)
             boot_cov = np.diag(boot_bse**2)

        if not hasattr(res_point, '_cache'):
             res_point._cache = {}

        for attr in ['bse', 'tvalues', 'pvalues', 'cov_params']:
             if attr in res_point._cache:
                  del res_point._cache[attr]

        res_point.cov_params_default = boot_cov
        res_point._cache['bse'] = boot_bse
        res_point._cache['cov_params'] = boot_cov
        
        # 避免除以零
        safe_bse = np.where(boot_bse < 1e-10, np.inf, boot_bse)
        tvalues = res_point.params / safe_bse
        res_point._cache['tvalues'] = tvalues
        
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))
        res_point._cache['pvalues'] = pvalues

        print(f"Bootstrap estimation completed with {len(boot_params)} successful iterations.")
        return res_point