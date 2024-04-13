# Codefest

Goal: Make guesses based on a data set


- We will preprocess a CSV file:
  - label each column data type
  - Remove data that never changes
- We will ask For a "guess" column
- make a guess:  
  - We go to each row, compare it to each other row, and find 3NN
  - We will have to check the column type we adding the points up
  - top N rows give us an output type. If Binary, make bindary guess. Otherwise average
