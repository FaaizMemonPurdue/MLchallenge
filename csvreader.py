import pandas as pd
import csv
import numpy as np
from constants import names
train_names = pd.read_csv('purdue-face-recognition-challenge-2024/train_small.csv')

def writeCSV(guesses, out_csv="guesses.csv"): #pass in Nx1 ndarr
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id","Category"])
        for i, elem in enumerate(guesses):
            writer.writerow([i, names[elem[0]]])

arr = np.array([[1], [2], [3], [4]])
writeCSV(arr)