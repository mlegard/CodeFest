import pandas as pd

def getFileName():
    fileName = input("Please enter a file name: ")
    while True:
        try:
            pd.read_csv(fileName)
            return fileName
        except:
            print("Please enter a valid file name in the same directory as this .py file: ")
            fileName = input()

def openFile(fileName):
    return pd.read_csv(fileName)

def printColumns(file):
    print("Available columns:")
    for col in file.columns:
        print(f"  - {col}")

def getValidColumn(file, fileName, guessColumn):
    printColumns(file)  # Print columns before asking for user input
    while True:
        chosenCol = input("Please enter a valid column from file " + fileName + " ")
        if chosenCol.lower() in [colName.lower() for colName in file.columns if colName.lower() != guessColumn.lower()]:
            return chosenCol
        else:
            print("Invalid column name. Please try again.")

def getGuessColumn(file, fileName):
    print("What guess column?")
    return getValidColumn(file, fileName, "")

def removeColumns(originalFile, fileName, guessColumn):
    newFile = originalFile
    while True:
        userInput = input("Remove a column? (Y/N) ")
        if userInput.lower() == 'n':
            return newFile
        newFile = removeColumn(newFile, fileName, guessColumn)
        print(newFile.head())

def removeColumn(file, fileName, guessColumn):
    chosenCol = getValidColumn(file, fileName, guessColumn)
    newFile = file.drop(chosenCol, axis=1)
    return newFile
