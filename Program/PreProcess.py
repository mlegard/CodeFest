import pandas as pd

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
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(variable_types.items(), columns=['Attribute', 'Data Type'])
    return df

