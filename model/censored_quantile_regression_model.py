import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm

class CensoredQuantileRegressionModel(GenericLikelihoodModel):
    """
    設限分位數迴歸模型 (Censored Quantile Regression Model, CQR)
    
    這是一個基於非對稱拉普拉斯分佈 (ALD) 概似函數的實作。
    適用於應變數在某些邊界處被設限 (Censored) 的情況（例如：評分最低 0 分，最高 1000 分）。
    """
    def __init__(self, endog, exog, tau=0.5, left=-np.inf, right=np.inf, **kwargs):
        # 確保資料格式正確
        endog = np.asarray(endog).flatten()
        exog = np.asarray(exog)
        super(CensoredQuantileRegressionModel, self).__init__(endog, exog, **kwargs)
        self.tau = tau
        self.left = left
        self.right = right
        
        if not 0 < tau < 1:
            raise ValueError("分位數 tau 必須在 0 和 1 之間。")

    def check_loss(self, u):
        """分位數損失函數 ρ_τ(u)"""
        return u * (self.tau - (u < 0).astype(float))

    def _ald_logpdf(self, u, sigma):
        """ALD 的對數機率密度函數 (Log PDF)"""
        sigma = np.maximum(sigma, 1e-10) # 避免除以零
        # log f(u) = log(τ(1-τ)/σ) - ρ_τ(u)/σ
        const_term = np.log(self.tau * (1 - self.tau)) - np.log(sigma)
        exp_term = -self.check_loss(u) / sigma
        return const_term + exp_term

    def _ald_logcdf(self, u, sigma):
        """ALD 的對數累積分佈函數 (Log CDF)"""
        sigma = np.maximum(sigma, 1e-10)
        z = u / sigma
        logcdf = np.zeros_like(u)
        
        mask_neg = u < 0
        mask_pos = ~mask_neg
        
        # u < 0 時: F(u) = τ * exp((1-τ)*u/σ)
        if np.any(mask_neg):
            logcdf[mask_neg] = np.log(self.tau) + (1 - self.tau) * z[mask_neg]
            
        # u >= 0 時: F(u) = 1 - (1-τ) * exp(-τ*u/σ)
        # 使用 log1p(x) = log(1+x) 來計算 log(1 - val) 以提高精度
        if np.any(mask_pos):
            val = (1 - self.tau) * np.exp(-self.tau * z[mask_pos])
            logcdf[mask_pos] = np.log1p(-val)
            
        return logcdf

    def _ald_logsf(self, u, sigma):
        """ALD 的對數生存函數 (Log SF = Log(1 - CDF))"""
        sigma = np.maximum(sigma, 1e-10)
        z = u / sigma
        logsf = np.zeros_like(u)
        
        mask_neg = u < 0
        mask_pos = ~mask_neg
        
        # u >= 0 時: SF(u) = (1-τ) * exp(-τ*u/σ)
        if np.any(mask_pos):
            logsf[mask_pos] = np.log(1 - self.tau) - self.tau * z[mask_pos]
            
        # u < 0 時: SF(u) = 1 - τ * exp((1-τ)*u/σ)
        if np.any(mask_neg):
            val = self.tau * np.exp((1 - self.tau) * z[mask_neg])
            logsf[mask_neg] = np.log1p(-val)
            
        return logsf

    def nloglikeobs(self, params):
        """負對數概似函數 (用於最小化)"""
        beta = params[:-1]
        sigma = np.abs(params[-1]) # 確保尺度參數為正
        
        mu = np.dot(self.exog, beta)
        resid = self.endog - mu
        
        # 定義不同區域的遮罩 (Boolean Masks)
        left_mask = (self.endog <= self.left)
        right_mask = (self.endog >= self.right)
        uncensored_mask = ~(left_mask | right_mask)
        
        ll = np.zeros_like(self.endog)
        
        # 1. 未設限區域：貢獻 PDF
        if np.any(uncensored_mask):
            ll[uncensored_mask] = self._ald_logpdf(resid[uncensored_mask], sigma)
            
        # 2. 左設限區域：貢獻 CDF (小於等於邊界的機率)
        if np.any(left_mask):
            resid_left = self.left - mu[left_mask]
            ll[left_mask] = self._ald_logcdf(resid_left, sigma)
            
        # 3. 右設限區域：貢獻 SF (大於等於邊界的機率)
        if np.any(right_mask):
             resid_right = self.right - mu[right_mask]
             ll[right_mask] = self._ald_logsf(resid_right, sigma)
             
        return -ll

    def fit(self, start_params=None, maxiter=1000, maxfun=1000, **kwargs):
        # 自動生成初始猜測值
        if start_params is None:
            try:
                # 嘗試用一般分位數迴歸 (QuantReg) 的結果當起點
                qr_model = sm.QuantReg(self.endog, self.exog)
                qr_res = qr_model.fit(q=self.tau)
                
                # 用殘差的 MAD 來估計初始 sigma
                resid = self.endog - qr_res.predict(self.exog)
                sigma_init = np.median(np.abs(resid - np.median(resid))) / 0.6745
                start_params = np.append(qr_res.params, np.maximum(sigma_init, 0.1))
            except:
                # 如果失敗，退回用 OLS 當起點
                ols_res = sm.OLS(self.endog, self.exog).fit()
                start_params = np.append(ols_res.params, np.std(ols_res.resid))

        print(f"開始擬合 CQR 模型 (tau={self.tau})...")
        # 使用 BFGS 優化算法，通常對這類問題效果不錯
        return super(CensoredQuantileRegressionModel, self).fit(
            start_params=start_params, 
            maxiter=maxiter, 
            maxfun=maxfun,
            method='bfgs',
             **kwargs
        )

# # =========================================
# # 模擬與測試
# # =========================================
# if __name__ == "__main__":
#     np.random.seed(42)
#     n_samples = 500
    
#     # 1. 生成自變數 X
#     x1 = np.random.normal(0, 1, n_samples)
#     x2 = np.random.binomial(1, 0.5, n_samples)
#     X = np.column_stack([np.ones(n_samples), x1, x2]) # 加上截距項
    
#     # 2. 真實參數 (假設 tau=0.5 時的真實關係)
#     true_beta = [50, 10, -5] 
    
#     # 3. 生成潛在變數 Y* (Latent Variable)
#     # 這裡用常態分佈誤差來生成，看看 ALD 模型能否還原它 (通常可以近似)
#     y_latent = np.dot(X, true_beta) + np.random.normal(0, 20, n_samples) 
    
#     # 4. 施加設限 (Censoring)
#     L, R = 30, 80
#     y_censored = np.clip(y_latent, L, R)
    
#     print(f"資料設限情況: 左側 {(y_censored==L).sum()} 筆, 右側 {(y_censored==R).sum()} 筆, 總共 {n_samples} 筆")

#     # 5. 擬合模型 (嘗試 tau = 0.5)
#     tau_to_fit = 0.5
#     model = CensoredQuantileRegressionModel(y_censored, X, tau=tau_to_fit, left=L, right=R)
#     res = model.fit()

#     print(res.summary())
    
#     print("\n真實參數:", true_beta)
#     print("估計參數:", res.params[:-1]) # 最後一個參數是 sigma
#     print("估計 Sigma:", res.params[-1])