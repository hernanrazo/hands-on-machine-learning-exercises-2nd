import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.model_selection import train_test_split

data = pd.read_csv('housing.csv')


print(data.info())

def test_set_check(identifier, ratio, hash):

    hash_value = hash(np.int64(identifier)).digest()[-1] < 256 * ratio

    return hash_value


def split_dataset(data, ratio, id_col, hash = hashlib.md5):

    ids = data[id_col]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, ratio, hash))
    new_data = data.loc[~in_test_set], data.loc[in_test_set]

    return new_data

housing_with_id = data.reset_index()
train_set, test_set = split_dataset(housing_with_id, 0.2, 'index')

housing_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
train_set, test_set = split_dataset(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42)

print("test: " + str(len(test_set)))
print("train: " + str(len(train_set)))
