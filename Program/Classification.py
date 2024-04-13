import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
def makeGuesses(processed_file, guess_column):

    # Split data into features and target
    y = processed_file[guess_column]
    X = processed_file.drop(guess_column, axis=1)

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
    r2 = r2_score(result[0], result[1])
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score = {r2}")




