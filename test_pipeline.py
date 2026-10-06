import pandas as pd
import numpy as np
from modules.preprocessing import run_pipeline_with_split

df = pd.read_csv('data/sample_dataset.csv')
num_cols = ['Age', 'Income', 'Tenure', 'Monthly_Expense', 'Credit_Score']
nom_cols = ['Gender', 'City', 'Occupation']
ord_cols = ['Education_Level', 'Satisfaction']
ord_dict = {
    'Education_Level': ['High School', 'Bachelor', 'Master', 'Doctorate'],
    'Satisfaction': ['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
}
fe_config = {
    'enable_log_income': True,
    'enable_expense_ratio': True,
    'enable_net_savings': True,
    'enable_age_binning': True
}

res = run_pipeline_with_split(
    df=df,
    numerical_cols=num_cols,
    nominal_cols=nom_cols,
    ordinal_cols=ord_cols,
    ordinal_categories_dict=ord_dict,
    test_size=0.2,
    random_state=42,
    scaler_type='standard',
    outlier_strategy='cap',
    fe_config=fe_config
)

print('=== 1. UKURAN X_TRAIN DAN X_TEST ===')
print('Ukuran X_train mentah:', res['X_train_raw'].shape)
print('Ukuran X_test mentah:', res['X_test_raw'].shape)
print('Ukuran X_train terproses:', res['X_train_proc'].shape)
print('Ukuran X_test terproses:', res['X_test_proc'].shape)

print('\n=== 2. JUMLAH FITUR SEBELUM DAN SESUDAH PREPROCESSING ===')
print('Jumlah fitur sebelum preprocessing (Data Mentah):', res['X_train_raw'].shape[1])
print('Jumlah fitur sesudah preprocessing (Pipeline Scikit-Learn):', res['X_train_proc'].shape[1])
print('Fitur yang dihasilkan FeatureEngineer:', [f['Nama Fitur'] for f in res['created_features']])

print('\n=== 3. VERIFIKASI NILAI MISSING SETELAH PIPELINE ===')
print('Jumlah missing values pada X_train_proc:', res['train_missing_count'])
print('Jumlah missing values pada X_test_proc:', res['test_missing_count'])

print('\n=== 4. PARAMETER PREPROCESSING YANG DIPELAJARI DARI TRAINING ===')
for param_name, param_val in res['learned_params'].items():
    print(f'\n-> {param_name}:')
    if isinstance(param_val, dict):
        for k, v in list(param_val.items())[:4]:
            print(f'   {k}: {v}')
        if len(param_val) > 4:
            print(f'   ... ({len(param_val) - 4} item lainnya)')

print('\n=== 5. BUKTI PIPELINE MELAKUKAN TRANSFORM PADA X_TEST TANPA REFIT ===')
test_transform_check = res['pipeline'].transform(res['X_test_raw'])
print('Eksekusi pipeline.transform(X_test) sukses!')
print('Dimensi hasil transform X_test:', test_transform_check.shape)
print('Apakah sama persis dengan X_test_proc?:', np.allclose(test_transform_check, res['X_test_proc'].values))
