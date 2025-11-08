import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


class MultipleLinearRegressionModel:
    def __init__(self, endog, exog, **kwargs):
        """
        初始化 OLS 多元線性回歸模型
        
        Parameters:
        -----------
        endog : array-like
            因變數 (y)
        exog : array-like
            自變數矩陣 (X)
        """
        self.endog = endog
        self.exog = exog
        self.model = None
        self.results = None

    def fit(self, **kwargs):
        """
        擬合 OLS 模型
        
        Returns:
        --------
        results : RegressionResults
            擬合結果
        """
        self.model = OLS(self.endog, self.exog)
        self.results = self.model.fit(**kwargs)
        return self.results

    def predict(self, exog=None):
        """
        預測新數據
        
        Parameters:
        -----------
        exog : array-like, optional
            新的自變數矩陣，若為 None 則使用訓練數據
            
        Returns:
        --------
        predictions : array
            預測值
        """
        if self.results is None:
            raise ValueError("模型尚未擬合，請先調用 fit() 方法")
        
        if exog is None:
            exog = self.exog
        
        return self.results.predict(exog)

    def summary(self):
        """
        顯示模型摘要
        
        Returns:
        --------
        summary : Summary
            模型摘要統計
        """
        if self.results is None:
            raise ValueError("模型尚未擬合，請先調用 fit() 方法")
        
        return self.results.summary()

    @property
    def params(self):
        """獲取迴歸係數"""
        if self.results is None:
            raise ValueError("模型尚未擬合，請先調用 fit() 方法")
        return self.results.params

    @property
    def rsquared(self):
        """獲取 R-squared"""
        if self.results is None:
            raise ValueError("模型尚未擬合，請先調用 fit() 方法")
        return self.results.rsquared

    @property
    def rsquared_adj(self):
        """獲取 Adjusted R-squared"""
        if self.results is None:
            raise ValueError("模型尚未擬合，請先調用 fit() 方法")
        return self.results.rsquared_adj