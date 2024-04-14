
import pandas as pd
def getFileName():
    fileName = input("Please enter a file name: ")
    while (True):
        try:
            pd.read_csv(fileName)
            return fileName
        except:
            print("Please enter a valid file name in the same directory as this .py file: ")
            fileName = input()
def openFile(fileName):
    return pd.read_csv(fileName)



def getValidColumn(file,fileName,guessColumn):
    while (True):
        chosenCol = input("Please enter a valid column from file " + fileName+ " ")
        for colName in file.columns:
            if (colName.lower() == chosenCol.lower() and chosenCol.lower() != guessColumn.lower()):
                return colName


def getGuessColumn(file,fileName):
    print("What guess column? (datatype of target variable must be binomial or continuous)")
    return getValidColumn(file,fileName,"")


def removeColumns(originalFile, fileName,guessColumn):

    newFile = originalFile
    while(True):
        userInput = input("Remove a column? (Y/N)")
        if(userInput == 'N' or userInput == 'n'): return newFile
        newFile = removeColumn(newFile,fileName,guessColumn)
        print(newFile.head())


def removeColumn(file,fileName,guessColumn):
    chosenCol = getValidColumn(file,fileName,guessColumn)
    newFile =file.drop(chosenCol,axis =1)
    return newFile

