import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def processFile(data):
    
    for col in data.select_dtypes(include=['float64', 'int64']):
        data[col].fillna(data[col].mean(), inplace=True)

    # Encode categorical data
    for col in data.select_dtypes(include=['object']):
        if data[col].nunique() <= 1000:  # arbitrary cutoff for categorical vs nominal
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        else:
            data = pd.get_dummies(data, columns=[col], drop_first=True)  # One-hot encoding

    return data



