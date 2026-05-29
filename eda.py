import pandas as pd
import os
import joblib

from sklearn.preprocessing import LabelEncoder

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/car_price.csv')

df.drop(
    columns=['Unnamed: 0'],
    errors='ignore',
    inplace=True
)

print(df.head())

print('\nInfo\n')
print(df.info())

print('\nNull Values\n')
print(df.isnull().sum())

print('\nDuplicates :', df.duplicated().sum())


# ---------------- CLEANING ----------------

df.drop_duplicates(inplace=True)

# Missing Value Handling

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    else:

        df[col] = df[col].fillna(
            df[col].median()
        )

# ---------------- ENCODING ----------------

encoders = {}

categorical_cols = [
    'car_name',
    'brand',
    'model',
    'seller_type',
    'fuel_type',
    'transmission_type'
]

for col in categorical_cols:

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------- SAVE CLEANED DATA ----------------

df.to_csv(
    'data/cleaned_car_price.csv',
    index=False
)
os.makedirs('models', exist_ok = True)

joblib.dump(
    encoders,
    'models/encoders.pkl'
)

print('\nCleaning Completed Successfully')