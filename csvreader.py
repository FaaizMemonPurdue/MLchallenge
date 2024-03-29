import pandas as pd
import csv
import numpy as np
from constants import names

def readLabelCSV(label_path):
    return pd.read_csv(label_path)

def writeGuessCSV(guesses, out_csv="guesses.csv"): #pass in Nx1 ndarr
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id","Category"])
        for i, elem in enumerate(guesses):
            writer.writerow([i, names[elem[0]]])

# samp_arr = np.array([[1], [2], [3], [4]])
# writeGuessCSV(samp_arr, samp_csv)