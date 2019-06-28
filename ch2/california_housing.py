import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('housing.csv')


print(data.info())

def split_dataset(data, ratio):

    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    test_data = data.iloc[test_indices]
    train_data = data.iloc[train_indices]

    return test_data, train_data


train_set, test_set = split_dataset(data, 0.2)
print("test: " + str(len(test_set)))
print("train: " + str(len(train_set)))
