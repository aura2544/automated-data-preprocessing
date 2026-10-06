import pandas as pd
import numpy as np

def generate_sample_data():
    np.random.seed(42)
    n = 1000

    # 1. Numerical Features (5)
    # Age: 20 - 65, with some outliers
    age = np.random.normal(38, 10, n).round()
    age = np.clip(age, 18, 75)
    # Add a few outliers for Age
    age[10:15] = [88, 92, 95, 89, 94]

    # Income: 3,000,000 to 25,000,000 IDR, skewed, with outliers
    income = np.random.exponential(scale=6000000, size=n) + 3500000
    income = np.round(income, -4)
    # Add high outliers for Income
    income[25:37] = [45000000, 52000000, 68000000, 75000000, 82000000, 95000000, 
                     58000000, 62000000, 70000000, 79000000, 88000000, 99000000]

    # Tenure: 1 to 15 years
    tenure = np.random.poisson(lam=4, size=n)
    tenure = np.clip(tenure, 0, 20)
    tenure[50:53] = [22, 24, 25]  # Outliers

    # Monthly_Expense: correlates somewhat with income
    monthly_expense = (income * np.random.uniform(0.3, 0.65, size=n)).round(-3)

    # Credit_Score: 350 to 850
    credit_score = np.random.normal(680, 75, n).round()
    credit_score = np.clip(credit_score, 350, 850)

    # 2. Nominal Features (3)
    gender = np.random.choice(['Male', 'Female'], size=n, p=[0.52, 0.48])
    city = np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Yogyakarta'], size=n, p=[0.35, 0.25, 0.18, 0.12, 0.10])
    occupation = np.random.choice(['Software Engineer', 'Data Analyst', 'Marketing', 'Teacher', 'Doctor', 'Entrepreneur'], size=n)

    # 3. Ordinal Features (2)
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], size=n, p=[0.20, 0.50, 0.23, 0.07])
    satisfaction = np.random.choice(['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'], size=n, p=[0.08, 0.15, 0.27, 0.35, 0.15])

    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Tenure': tenure,
        'Monthly_Expense': monthly_expense,
        'Credit_Score': credit_score,
        'Gender': gender,
        'City': city,
        'Occupation': occupation,
        'Education_Level': education_level,
        'Satisfaction': satisfaction
    })

    # Total missing values target: exactly 35 missing values across columns
    # Age: 10 missing
    df.loc[100:109, 'Age'] = np.nan
    # Income: 5 missing
    df.loc[120:124, 'Income'] = np.nan
    # Tenure: 4 missing
    df.loc[130:133, 'Tenure'] = np.nan
    # Monthly_Expense: 6 missing
    df.loc[140:145, 'Monthly_Expense'] = np.nan
    # Credit_Score: 3 missing
    df.loc[150:152, 'Credit_Score'] = np.nan
    # City: 3 missing
    df.loc[160:162, 'City'] = np.nan
    # Satisfaction: 4 missing
    df.loc[170:173, 'Satisfaction'] = np.nan

    # Check total missing: 10 + 5 + 4 + 6 + 3 + 3 + 4 = 35 missing values
    return df

if __name__ == '__main__':
    df = generate_sample_data()
    df.to_csv('data/sample_dataset.csv', index=False)
    print("Sample dataset successfully created!")
    print(f"Shape: {df.shape}")
    print(f"Total missing values: {df.isna().sum().sum()}")
    print("\nMissing per column:")
    print(df.isna().sum())
