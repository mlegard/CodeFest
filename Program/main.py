#!/usr/bin/env python
import Classification
import PreProcess
import UserInput


def main():
    print("Welcome to our data set predictor!")
    fileName = UserInput.getFileName()
    originalFile = UserInput.openFile(fileName)
    guessColumn = UserInput.getGuessColumn(originalFile,fileName)
    colRemovedFile = UserInput.removeColumns(originalFile,fileName,guessColumn)
    processedFile = PreProcess.processFile(colRemovedFile)
    print(processedFile)
    results = Classification.makeGuesses(processedFile,guessColumn)
    Classification.printAcurracy(results)

if __name__ == "__main__":
    main()