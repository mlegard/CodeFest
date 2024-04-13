#!/usr/bin/env python
import Classification
import PreProcess
import UserInput


def main():
    print("Welcome to our data set predictor!")
    fileName = UserInput.getFileName()
    originalFile = UserInput.openFile(fileName)
    processedFile = PreProcess.processFile(originalFile)
    guessColumn = UserInput.getGuessColumn()
    results = Classification.makeGuesses(processedFile,guessColumn)
    Classification.printAccuracy(processedFile,results)
    #
    #
    #

if __name__ == "__main__":
    main()