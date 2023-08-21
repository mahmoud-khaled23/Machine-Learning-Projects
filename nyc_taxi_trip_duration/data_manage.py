import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# load_dataset():
"""
TAKES ==> path to the datasets, train dataset name, and validation dataset name

RETURNS ==> train dataset (X), val dataset (y)
"""

def load_dataset(root='', path='datasets/split/', train_dataset='train.csv', val_dataset='val.csv'):
    # root = '../'
    X = pd.read_csv(root + path + train_dataset)
    y = pd.read_csv(root + path + val_dataset)

    return X, y


# split_data():
"""
if apply manually splitting: (X) is train dataset, 
    (y) is val dataset

TAKES ==> train dataset (X), val dataset (y), train_test_split -bool-, to_numpy() -bool-
RETURNS ==> 
"""
def split_data(X, y, toNumpy=False):
    # split the data manually, and this is the suitable way for our data

    X_train = X.iloc[:, :-1]
    y_train = X.iloc[:, -1]

    X_val = y.iloc[:, :-1]
    y_val = y.iloc[:, -1]

    if toNumpy:
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

        X_val = X_val.to_numpy()
        y_val = y_val.to_numpy()

    # train input data (X_train), val input data (X_val)
    # train target value (y_train), val target value (y_val)
    return X_train, X_val, y_train, y_val

def cols_to_drop(X, y, cols, is_series=False):
    if is_series:
        X = X.drop([cols], axis=1)
        y = y.drop([cols], axis=1)
    else:
        X = X.drop(cols, axis=1)
        y = y.drop(cols, axis=1)

    return X, y

# get_cat_col():
"""
TAKES ==> Pandas DataFrame of train Input Data
RETURNS ==> categorical columns in the DataFrame
"""
def cat_cols_to_encode(X, y):
    # ATTENTION ===> ----- i choose just 'store_and_fwd_flag' column ----- <===
    # df_cat = df.select_dtypes(include=['object'])
    X_cat = X['store_and_fwd_flag']
    y_cat = y['store_and_fwd_flag']
    X_cat_names: str = ''

    is_series = False
    if X_cat.size == X_cat.shape[0]:
        X_cat_names = X_cat.name
        is_series = True
    else:
        X_cat_names = X_cat.columns

    return X_cat, y_cat, X_cat_names, is_series


# get_numerical_col():
"""
TAKES ==> Pandas DataFrame of train Input Data
RETURNS ==> numerical columns in the DataFrame
"""
def get_numeric_col(df):
    df_num = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])

    return df_num

def obj_to_datetime(X, y):
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    y['pickup_datetime'] = pd.to_datetime(y['pickup_datetime']).values

    X_date = X['pickup_datetime'].values
    y_date = y['pickup_datetime'].values
    # dfd = y['pickup_datetime'].values

    X_date = X_date.reshape(-1, 1)
    y_date = y_date.reshape(-1, 1)

    # print(type(X.iloc[0, 1]))
    # print(type(y.iloc[0, 1]))
    # print(type(dfd[0]))
    return X_date, y_date


# encode_data():
"""
TAKES ==> train Input Data (X_train), val Input Data (X_val)
            categorical columns (columns), chosen encoder (encoder) default(OneHotEncoder())
RETURNS ==> encoded data with the applied encoder (encoded) -array or list (idk)- 
            (Transformed array) in the documentation 
"""
def encode_data(features, is_series=False, encoder=OneHotEncoder()):
    # transformer = make_column_transformer((OneHotEncoder(), columns),
    #                                       remainder='passthrough')
    if is_series:
        features = features.reshape(-1, 1)

    OHE = OneHotEncoder(sparse_output=False)

    encoded = OHE.fit_transform(features)
    print(f'fit returns: {type(encoded)}')

    return encoded


def concat_encoded_cols(df, features_names, transformed, categories):
    df = df.loc[:, ~df.columns.isin([features_names])]

    n_cat_feats = 0
    for cat in categories:
        for cat_transformed_feature_name, idx in zip(cat, range(n_cat_feats, n_cat_feats+len(cat))):
            df[cat_transformed_feature_name] = transformed[:, idx]
            # print(f'categorical feature of categories: {cat_transformed_feature_name}')
            # print(f'value of x: {idx}')

        n_cat_feats = n_cat_feats + len(cat)

    return df

def datasets_to_numpy_array(X_train, X_val, y_train, y_val):
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    return X_train, X_val, y_train, y_val

def cats_to_numpy_array(cat_cols):
    cat_cols = cat_cols.to_numpy()

    return cat_cols


# scale_data():
"""
TAKES ==> train Input Data (X_train), val Input Data (X_val), chosen scaler (scaler) default(MinMaxScaler())
RETURNS ==> scaled train Input Data with the applied scaler (X_train)
            scaled val Input Data with the applied scaler (X_val)
"""
def scale_data(X_train, X_val, y_train, y_val, scaler=MinMaxScaler()):
    processor = scaler
    X_train = processor.fit_transform(X_train)
    y_train = y_train.reshape(-1, 1)
    y_train = processor.fit_transform(y_train)

    # in the X_val we used transform not fit_transform
    X_val = processor.fit_transform(X_val)
    y_val = y_val.reshape(-1, 1)
    y_val = processor.fit_transform(y_val)

    return X_train, X_val, y_train, y_val
