
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



def getValidColumn(file,fileName):
    while (True):
        chosenCol = input("Please enter a valid column from file " + fileName+ " ")
        for colName in file.columns:
            if (colName.lower() == chosenCol.lower()):
                return colName


def getGuessColumn(file,fileName):
    print("What guess column?")
    return getValidColumn(file,fileName)


def removeColumns(originalFile, fileName):
    wantsToRemove = True
    newFile = originalFile
    while(True):
        userInput = input("Remove a column? (Y/N)")
        if(userInput == 'N' or userInput == 'n'): return newFile
        newFile = removeColumn(newFile,fileName)
        print(newFile.head())


def removeColumn(file,fileName):
    chosenCol = getValidColumn(file,fileName)
    newFile =file.drop(chosenCol,axis =1)
    return newFile

