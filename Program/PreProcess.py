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


def preprocess_for_knn(data, guess_column):
    """
    Preprocess the data for kNN classification by handling missing values,
    encoding categorical variables, and scaling numerical features.

    Parameters:
    data (DataFrame): The input data.
    guess_column (str): The target variable column name.

    Returns:
    X (DataFrame): Processed feature matrix.
    y (Series): Target variable.
    """
    # Separate the features and the target variable
    X = data.drop(columns=[guess_column])
    y = data[guess_column]

    # Define which columns are categorical
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int', 'float']).columns

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Scale data
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One hot encode categorical variables
    ])

    # Combine preprocessing steps into one ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply the preprocessing pipeline to the feature data
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y


