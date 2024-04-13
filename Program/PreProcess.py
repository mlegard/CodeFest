import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def processFile(originalFile):
    return None

def categorize_variable_types(data):
    variable_types = {}
    for column in data.columns:
        unique_values = data[column].nunique()
        data_type = data[column].dtype

        # Automatically categorize variable types
        if unique_values == 2:  # Binary
            variable_types[column] = 'binary'
        elif data_type == 'object':  # Nominal
            variable_types[column] = 'nominal'
        elif unique_values > 2 and unique_values < 10:  # Nominal (or categorical with fewer categories)
            variable_types[column] = 'nominal'
        else:  # Continuous
            variable_types[column] = 'continuous'
    return variable_types
