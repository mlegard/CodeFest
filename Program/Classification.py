import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def pickClassification(processed_file, guess_column):
    guessType = processed_file[guess_column].dtype
    print(guessType)#NOT PRINTING THE TYPE, but instead it prints int64 ect
    #if(guessType == "binary" or guessType == "nominal"): return knn_classification(processed_file, guess_column)
    #if(guessType == "continuous"): return linearRegression_classification(processed_file,guess_column)
    print("GOT HERE")
def knn_classification(processed_file, guess_column):
    # Separate features and target variable
    X = processed_file.drop(columns=[guess_column])
    y = processed_file[guess_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    # Initialize the kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    
    # Train the classifier
    knn_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_classifier.predict(X_test)

    return y_test, y_pred
    
def linearRegression_classification(processed_file, guess_column):
    # File info
    processed_file.info()
    processed_file.head()

    # CHeck if column exists
    if guess_column not in processed_file.columns:
        raise ValueError(f"Column '{guess_column}' not found in the DataFrame.")

    # Split data into features and target
    X = processed_file.drop(guess_column, axis=1)
    y = processed_file[guess_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make prediction
    y_pred = model.predict(X_test)

    return y_test, y_pred

def printAcurracy(result):
    mse = mean_squared_error(result[0], result[1])
    rmse = np.sqrt(mse)
    r2 = r2_score(result[0], result[1])
    print(f"Mean Squared Error: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R^2 Score = {r2}")

def plotModelDiagnostics(actual, predicted):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=actual, y=predicted)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')

    plt.subplot(1, 2, 2)
    residuals = actual - predicted
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')

    plt.tight_layout()
    plt.show()
