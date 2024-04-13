#!/usr/bin/env python
import Classification
import PreProcess
import UserInput
from Codefest.Program import Methods


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
    Classification.plotModelDiagnostics(results[0],results[1])

if __name__ == "__main__":
    main()