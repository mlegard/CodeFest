
import pandas as pd
def getFileName():
    fileName = input("Please enter a file name: ")
    while (True):
        try:
            pd.read_csv(fileName)
            return fileName
        except:
            print("Please enter a valid file name in the same directory as this .py file")
            fileName = input()
def openFile(fileName):
    return pd.read_csv(fileName)


def getGuessColumn(file,fileName):
    while(True):
        chosenCol = input("Please enter a valid column from file "+fileName)
        for colName in file.columns:
            if(colName == chosenCol):
                return colName

