import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible Transformer for Automated Feature Engineering.
    Inherits from BaseEstimator and TransformerMixin.
    
    CRITICAL FOR ZERO DATA LEAKAGE:
    - Integrated directly as the first stage of the Scikit-Learn Pipeline.
    - fit(X_train) learns column structures and prepares transformations strictly from training data.
    - transform() applies row-wise transformations preserving missing values for downstream SimpleImputer.
    - Generates 4 required features:
      1. Log_Income (Log Transformation for skewed Income)
      2. Expense_to_Income_Ratio (Financial expense ratio)
      3. Net_Savings (Disposable monthly savings)
      4. Age_Group (Demographic age binning)
    """
    def __init__(self,
                 enable_log_income: bool = True,
                 enable_expense_ratio: bool = True,
                 enable_net_savings: bool = True,
                 enable_age_binning: bool = True,
                 income_col: str = 'Income',
                 expense_col: str = 'Monthly_Expense',
                 age_col: str = 'Age'):
        self.enable_log_income = enable_log_income
        self.enable_expense_ratio = enable_expense_ratio
        self.enable_net_savings = enable_net_savings
        self.enable_age_binning = enable_age_binning
        self.income_col = income_col
        self.expense_col = expense_col
        self.age_col = age_col
        self.created_features_ = []
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        created = []

        # 1. Log Transformation for skewed income
        if self.enable_log_income and self.income_col in X_df.columns:
            created.append({
                'Nama Fitur': 'Log_Income',
                'Tipe': 'Numerical (Log Transform)',
                'Formula': f'log1p(max(0, {self.income_col}))',
                'Tujuan / Alasan': 'Mengurangi skewness (kecondongan) distribusi pendapatan agar mendekati normal'
            })

        # 2. Ratio Feature: Expense to Income Ratio
        if self.enable_expense_ratio and self.expense_col in X_df.columns and self.income_col in X_df.columns:
            created.append({
                'Nama Fitur': 'Expense_to_Income_Ratio',
                'Tipe': 'Numerical (Financial Ratio)',
                'Formula': f'{self.expense_col} / {self.income_col}',
                'Tujuan / Alasan': 'Mengukur proporsi beban pengeluaran bulanan terhadap total pemasukan'
            })

        # 3. Combined Feature: Net Savings
        if self.enable_net_savings and self.income_col in X_df.columns and self.expense_col in X_df.columns:
            created.append({
                'Nama Fitur': 'Net_Savings',
                'Tipe': 'Numerical (Combined Difference)',
                'Formula': f'{self.income_col} - {self.expense_col}',
                'Tujuan / Alasan': 'Menghitung sisa likuiditas dana tabungan bersih per bulan'
            })

        # 4. Binning Feature: Age Group
        if self.enable_age_binning and self.age_col in X_df.columns:
            created.append({
                'Nama Fitur': 'Age_Group',
                'Tipe': 'Nominal (Binned Categories)',
                'Formula': f'cut({self.age_col}, bins=[0, 30, 50, 120])',
                'Tujuan / Alasan': 'Segmentasi demografis: Young Adult (<30), Middle-aged (30-50), Senior (>50)'
            })

        self.created_features_ = created
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Log Transformation (preserves NaN so downstream SimpleImputer handles it)
        if self.enable_log_income and self.income_col in X_df.columns:
            inc = pd.to_numeric(X_df[self.income_col], errors='coerce')
            log_inc = np.where(inc.isna(), np.nan, np.log1p(np.maximum(0, inc)))
            X_df['Log_Income'] = log_inc

        # 2. Ratio Feature: Expense to Income Ratio
        if self.enable_expense_ratio and self.expense_col in X_df.columns and self.income_col in X_df.columns:
            exp = pd.to_numeric(X_df[self.expense_col], errors='coerce')
            inc = pd.to_numeric(X_df[self.income_col], errors='coerce')
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = np.where((inc <= 0) | inc.isna(), np.nan, inc)
                ratio = np.where(exp.isna() | np.isnan(denom), np.nan, exp / denom)
                ratio = np.clip(ratio, 0.0, 10.0)
            X_df['Expense_to_Income_Ratio'] = ratio

        # 3. Combined Feature: Net Savings
        if self.enable_net_savings and self.income_col in X_df.columns and self.expense_col in X_df.columns:
            inc = pd.to_numeric(X_df[self.income_col], errors='coerce')
            exp = pd.to_numeric(X_df[self.expense_col], errors='coerce')
            net_save = np.where(inc.isna() | exp.isna(), np.nan, inc - exp)
            X_df['Net_Savings'] = net_save

        # 4. Binning Feature: Age Group (preserves NaN as np.nan so downstream SimpleImputer handles it)
        if self.enable_age_binning and self.age_col in X_df.columns:
            age = pd.to_numeric(X_df[self.age_col], errors='coerce')
            def categorize_age(val):
                if pd.isna(val):
                    return np.nan
                if val < 30:
                    return 'Young Adult'
                elif val <= 50:
                    return 'Middle-aged'
                else:
                    return 'Senior'
            X_df['Age_Group'] = age.apply(categorize_age)

        return X_df

    def get_feature_names_out(self, input_features=None):
        base_cols = list(input_features) if input_features is not None else list(self.feature_names_in_)
        added = []
        if self.enable_log_income and self.income_col in base_cols:
            added.append('Log_Income')
        if self.enable_expense_ratio and self.expense_col in base_cols and self.income_col in base_cols:
            added.append('Expense_to_Income_Ratio')
        if self.enable_net_savings and self.income_col in base_cols and self.expense_col in base_cols:
            added.append('Net_Savings')
        if self.enable_age_binning and self.age_col in base_cols:
            added.append('Age_Group')
        return np.array(base_cols + added)


def apply_custom_feature_engineering(df: pd.DataFrame, config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function to preview feature engineering outputs interactively in the UI.
    """
    fe = FeatureEngineer(
        enable_log_income=config.get('enable_log_income', True) if config else True,
        enable_expense_ratio=config.get('enable_expense_ratio', True) if config else True,
        enable_net_savings=config.get('enable_net_savings', True) if config else True,
        enable_age_binning=config.get('enable_age_binning', True) if config else True,
        income_col=config.get('income_col', 'Income') if config else 'Income',
        expense_col=config.get('expense_col', 'Monthly_Expense') if config else 'Monthly_Expense',
        age_col=config.get('age_col', 'Age') if config else 'Age'
    )
    fe.fit(df)
    transformed_df = fe.transform(df)
    summary_df = pd.DataFrame(fe.created_features_)
    return transformed_df, summary_df
