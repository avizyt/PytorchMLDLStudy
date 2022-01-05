import pandas as pd

file_path = '../data/superconduct/unique_m.csv'

data = pd.read_csv(file_path, sep=',', header=None)

# print(data.head())

print(data.shape)