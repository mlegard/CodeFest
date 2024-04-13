import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def processFile(data):
    # Handle missing values (example: numeric imputation)
    for col in data.select_dtypes(include=['float64', 'int64']):
        data[col].fillna(data[col].mean(), inplace=True)

    # Encode categorical data
    for col in data.select_dtypes(include=['object']):
        if data[col].nunique() <= 10:  # arbitrary cutoff for categorical vs nominal
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        else:
            data = pd.get_dummies(data, columns=[col], drop_first=True)  # One-hot encoding

    return data

