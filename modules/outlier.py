import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer for IQR-based outlier capping (Winsorization/Clipping).
    CRITICAL FOR DATA LEAKAGE PREVENTION:
    - fit() calculates Q1, Q3, and clipping bounds ONLY from the training dataset.
    - transform() applies these fixed bounds to train and test datasets.
    """
    def __init__(self, factor=1.5, handle_strategy='cap'):
        self.factor = factor
        self.handle_strategy = handle_strategy
        self.bounds_ = {}
        self.n_features_in_ = 0

    def fit(self, X, y=None):
        if self.handle_strategy == 'none':
            return self

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.n_features_in_ = X_df.shape[1]
        self.bounds_ = {}
        
        for col in X_df.columns:
            series = pd.to_numeric(X_df[col], errors='coerce').dropna()
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (self.factor * iqr)
                upper_bound = q3 + (self.factor * iqr)
                self.bounds_[col] = (lower_bound, upper_bound)
            else:
                self.bounds_[col] = (-np.inf, np.inf)
                
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        if self.handle_strategy == 'none' or not self.bounds_:
            return X if is_df else np.array(X)

        X_df = pd.DataFrame(X).copy()
        
        for col in X_df.columns:
            if col in self.bounds_:
                low, high = self.bounds_[col]
                X_df[col] = X_df[col].clip(lower=low, upper=high)
                
        return X_df if is_df else X_df.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"x{i}" for i in range(self.n_features_in_)])


def detect_outliers_iqr(df: pd.DataFrame, numerical_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Detects outliers across numerical columns using the IQR method.
    Returns a structured summary DataFrame.
    """
    summary = []
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
            
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        n = len(series)
        if n == 0:
            continue
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df[col])) * 100
        
        summary.append({
            'Atribut': col,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'Lower Bound': round(lower_bound, 2),
            'Upper Bound': round(upper_bound, 2),
            'Jumlah Outlier': outlier_count,
            'Persentase Outlier (%)': round(outlier_pct, 2)
        })
        
    return pd.DataFrame(summary)
