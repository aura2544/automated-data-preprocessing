import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from modules.outlier import IQROutlierCapper
from modules.feature_engineering import FeatureEngineer

def build_full_pipeline(
    numerical_cols: list,
    nominal_cols: list,
    ordinal_cols: list,
    ordinal_categories_dict: dict,
    num_impute_strategy: str = 'median',
    cat_impute_strategy: str = 'most_frequent',
    scaler_type: str = 'standard',
    outlier_strategy: str = 'cap',
    fe_config: dict = None
) -> tuple[Pipeline, list, list, list]:
    """
    Constructs an end-to-end Scikit-Learn Pipeline following the exact architectural requirement:
    
    RAW DATA
       ↓
    Train/Test Split (before pipeline)
       ↓
    Scikit-Learn Pipeline
       ↓
    Feature Engineering (FeatureEngineer: Log_Income, Expense_to_Income_Ratio, Net_Savings, Age_Group)
       ↓
    ColumnTransformer:
       - Numerical (Original + Engineered Numerical):
           SimpleImputer (median) → IQROutlierCapper (Train Bounds) → StandardScaler (Train μ & σ)
       - Nominal (Original + Engineered Age_Group):
           SimpleImputer (most_frequent) → OneHotEncoder (handle_unknown='ignore')
       - Ordinal:
           SimpleImputer (most_frequent) → OrdinalEncoder (explicit categories)
       ↓
    PROCESSED DATA (Train & Test)
    """
    fe_config = fe_config or {}
    enable_log_income = fe_config.get('enable_log_income', True)
    enable_expense_ratio = fe_config.get('enable_expense_ratio', True)
    enable_net_savings = fe_config.get('enable_net_savings', True)
    enable_age_binning = fe_config.get('enable_age_binning', True)
    income_col = fe_config.get('income_col', 'Income')
    expense_col = fe_config.get('expense_col', 'Monthly_Expense')
    age_col = fe_config.get('age_col', 'Age')

    # 1. Feature Engineering Stage
    fe_transformer = FeatureEngineer(
        enable_log_income=enable_log_income,
        enable_expense_ratio=enable_expense_ratio,
        enable_net_savings=enable_net_savings,
        enable_age_binning=enable_age_binning,
        income_col=income_col,
        expense_col=expense_col,
        age_col=age_col
    )

    # 2. Determine effective columns following feature engineering
    effective_num_cols = list(numerical_cols)
    if enable_log_income and 'Log_Income' not in effective_num_cols:
        effective_num_cols.append('Log_Income')
    if enable_expense_ratio and 'Expense_to_Income_Ratio' not in effective_num_cols:
        effective_num_cols.append('Expense_to_Income_Ratio')
    if enable_net_savings and 'Net_Savings' not in effective_num_cols:
        effective_num_cols.append('Net_Savings')

    effective_nom_cols = list(nominal_cols)
    if enable_age_binning and 'Age_Group' not in effective_nom_cols:
        effective_nom_cols.append('Age_Group')

    effective_ord_cols = list(ordinal_cols)

    # 3. ColumnTransformer Sub-Pipelines
    transformers = []

    # Numerical Sub-Pipeline
    if effective_num_cols:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        num_steps = [
            ('imputer', SimpleImputer(strategy=num_impute_strategy)),
            ('outlier', IQROutlierCapper(handle_strategy=outlier_strategy)),
            ('scaler', scaler)
        ]
        transformers.append(('num', Pipeline(steps=num_steps), effective_num_cols))

    # Nominal Sub-Pipeline
    if effective_nom_cols:
        nom_steps = [
            ('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
        transformers.append(('nom', Pipeline(steps=nom_steps), effective_nom_cols))

    # Ordinal Sub-Pipeline
    if effective_ord_cols:
        categories_list = [ordinal_categories_dict[col] for col in effective_ord_cols if col in ordinal_categories_dict]
        ord_steps = [
            ('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value='Missing')),
            ('ordinal', OrdinalEncoder(
                categories=categories_list,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ]
        transformers.append(('ord', Pipeline(steps=ord_steps), effective_ord_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )

    # End-to-End Unified Pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', fe_transformer),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline, effective_num_cols, effective_nom_cols, effective_ord_cols


# Backwards compatibility alias
def build_preprocessing_pipeline(
    numerical_cols: list,
    nominal_cols: list,
    ordinal_cols: list,
    ordinal_categories_dict: dict,
    num_impute_strategy: str = 'median',
    cat_impute_strategy: str = 'most_frequent',
    scaler_type: str = 'standard',
    outlier_strategy: str = 'cap',
    fe_config: dict = None
):
    pipe, _, _, _ = build_full_pipeline(
        numerical_cols=numerical_cols,
        nominal_cols=nominal_cols,
        ordinal_cols=ordinal_cols,
        ordinal_categories_dict=ordinal_categories_dict,
        num_impute_strategy=num_impute_strategy,
        cat_impute_strategy=cat_impute_strategy,
        scaler_type=scaler_type,
        outlier_strategy=outlier_strategy,
        fe_config=fe_config
    )
    return pipe


def get_feature_names_from_full_pipeline(
    pipeline: Pipeline,
    effective_num_cols: list,
    effective_nom_cols: list,
    effective_ord_cols: list
) -> list:
    """
    Retrieves transformed feature names from fitted ColumnTransformer.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    try:
        names = list(preprocessor.get_feature_names_out())
        clean_names = []
        for name in names:
            for prefix in ['num__', 'nom__', 'ord__']:
                if name.startswith(prefix):
                    name = name[len(prefix):]
            clean_names.append(name)
        return clean_names
    except Exception:
        feature_names = []
        if 'num' in preprocessor.named_transformers_:
            feature_names.extend(effective_num_cols)
        if 'nom' in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_['nom'].named_steps.get('onehot')
            if ohe and hasattr(ohe, 'get_feature_names_out'):
                feature_names.extend(list(ohe.get_feature_names_out(effective_nom_cols)))
            else:
                feature_names.extend([f"nom_{i}" for i in range(len(effective_nom_cols))])
        if 'ord' in preprocessor.named_transformers_:
            feature_names.extend(effective_ord_cols)
        return feature_names


def run_pipeline_with_split(
    df: pd.DataFrame,
    numerical_cols: list,
    nominal_cols: list,
    ordinal_cols: list,
    ordinal_categories_dict: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = 'standard',
    outlier_strategy: str = 'cap',
    fe_config: dict = None,
    target_col: str = None
):
    """
    Executes train/test split on RAW DATA, fits unified pipeline ONLY on training data,
    and transforms train and test data separately to guarantee ZERO data leakage.
    """
    # 1. Train/Test Split on RAW DATA
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X = df.copy()
        y_train, y_test = None, None
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )

    # 2. Build Unified Pipeline (Feature Engineering + ColumnTransformer)
    pipeline, eff_num, eff_nom, eff_ord = build_full_pipeline(
        numerical_cols=numerical_cols,
        nominal_cols=nominal_cols,
        ordinal_cols=ordinal_cols,
        ordinal_categories_dict=ordinal_categories_dict,
        scaler_type=scaler_type,
        outlier_strategy=outlier_strategy,
        fe_config=fe_config
    )

    # 3. Fit ONLY on X_train (Zero Data Leakage!)
    pipeline.fit(X_train)

    # 4. Transform X_train and X_test without any refitting
    X_train_proc = pipeline.transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    # 5. Extract feature names
    feature_names = get_feature_names_from_full_pipeline(
        pipeline, eff_num, eff_nom, eff_ord
    )

    df_train_proc = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    df_test_proc = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

    # 6. Extract parameters learned strictly from training data
    preprocessor = pipeline.named_steps['preprocessor']
    learned_params = {}

    if 'num' in preprocessor.named_transformers_ and eff_num:
        num_pipe = preprocessor.named_transformers_['num']
        
        # Medians from SimpleImputer
        imputer = num_pipe.named_steps.get('imputer')
        if imputer and hasattr(imputer, 'statistics_'):
            learned_params['Numerical Median Imputation (dari Train Data)'] = {
                col: float(np.round(stat, 4)) for col, stat in zip(eff_num, imputer.statistics_)
            }
        
        # Outlier Bounds from IQROutlierCapper
        capper = num_pipe.named_steps.get('outlier')
        if capper and hasattr(capper, 'bounds_') and capper.bounds_:
            bound_dict = {}
            for idx, col in enumerate(eff_num):
                if col in capper.bounds_:
                    b = capper.bounds_[col]
                elif idx in capper.bounds_:
                    b = capper.bounds_[idx]
                else:
                    b = (-np.inf, np.inf)
                bound_dict[col] = (float(round(b[0], 2)), float(round(b[1], 2)))
            learned_params['Outlier Bounds (IQR dari Train Data)'] = bound_dict

        # Scaler Mean & Scale
        scaler = num_pipe.named_steps.get('scaler')
        if scaler and hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            learned_params['Scaler Mean (dari Train Data)'] = {
                col: float(np.round(val, 4)) for col, val in zip(eff_num, scaler.mean_)
            }
            learned_params['Scaler Scale / Std (dari Train Data)'] = {
                col: float(np.round(val, 4)) for col, val in zip(eff_num, scaler.scale_)
            }

    # Nominal categories learned
    if 'nom' in preprocessor.named_transformers_ and eff_nom:
        nom_pipe = preprocessor.named_transformers_['nom']
        ohe = nom_pipe.named_steps.get('onehot')
        if ohe and hasattr(ohe, 'categories_'):
            cat_dict = {}
            for col, cats in zip(eff_nom, ohe.categories_):
                cat_dict[col] = [str(c) for c in cats]
            learned_params['Nominal Categories Learned (dari Train Data)'] = cat_dict

    # Check zero missing values
    train_missing = int(df_train_proc.isna().sum().sum())
    test_missing = int(df_test_proc.isna().sum().sum())

    fe_step = pipeline.named_steps['feature_engineering']
    created_fe = getattr(fe_step, 'created_features_', [])

    return {
        'pipeline': pipeline,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_proc': df_train_proc,
        'X_test_proc': df_test_proc,
        'feature_names': feature_names,
        'learned_params': learned_params,
        'effective_num_cols': eff_num,
        'effective_nom_cols': eff_nom,
        'effective_ord_cols': eff_ord,
        'created_features': created_fe,
        'train_missing_count': train_missing,
        'test_missing_count': test_missing
    }
