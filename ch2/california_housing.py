import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('housing.csv')

print(' ')
print('printing data info:')
print(data.info())
print('=================================================================')


def test_set_check(identifier, ratio, hash):

    hash_value = hash(np.int64(identifier)).digest()[-1] < 256 * ratio

    return hash_value


def split_dataset(data, ratio, id_col, hash = hashlib.md5):

    ids = data[id_col]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, ratio, hash))
    new_data = data.loc[~in_test_set], data.loc[in_test_set]

    return new_data

def income_cat_proportions(data_set):

    income_cat = data_set['income_cat'].value_counts() / len(data_set)

    return income_cat


housing_with_id = data.reset_index()
print(' ')
print('new dataframe:')
print(housing_with_id.head())
print('=================================================================')

train_set, test_set = split_dataset(housing_with_id, 0.2, 'index')

housing_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
train_set, test_set = split_dataset(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42)


print(' ')
print('Printing testing and training set sizes:')
print("test: " + str(len(test_set)))
print("train: " + str(len(train_set)))
print('=================================================================')



data['income_cat'] = np.ceil(data['median_income'] / 1.5)
data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace = True)

income_split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in income_split.split(data, data['income_cat']):

    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]


train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42)

compare_proportions = pd.DataFrame({'Overall': income_cat_proportions(data),
    'Stratified': income_cat_proportions(strat_test_set),
    'Random': income_cat_proportions(test_set),}).sort_index()

compare_proportions['Rand. %Error'] = 100 * compare_proportions['Random'] / compare_proportions['Overall'] - 100
compare_proportions['Strat. %Error'] = 100 * compare_proportions['Stratified'] / compare_proportions['Overall'] - 100

print(' ')
print('comparing proportions:')
print(compare_proportions)
print('=================================================================')


for set in (strat_train_set, strat_test_set):

    set.drop(['income_cat'], axis = 1, inplace = True)

#data cleaning

print(' ')
print('starting data cleaning process...')
print(' ')

housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis = 1)].head()

median = housing['total_bedrooms'].median()
sample_incomplete_rows['total_bedrooms'].fillna(median, inplace = True)

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy = 'median')

housing_num = housing.drop('ocean_proximity', axis = 1)


imputer.fit(housing_num)

imputer.statistics_

x = imputer.transform(housing_num)

housing_tr = pd.DataFrame(x, columns = housing_num.columns, index = housing.index)
housing_tr.loc[sample_incomplete_rows.index.values]

housing_cat = housing[['ocean_proximity']]

try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

try:
    #line 63:














