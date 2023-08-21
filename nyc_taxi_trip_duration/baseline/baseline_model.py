from nyc_taxi_trip_duration.data_manage import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


def lin_reg(X_train, t_train):
    model = LinearRegression()
    model.fit(X_train, t_train)

    return model


def eval_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return r2


if __name__ == '__main__':
    # 1- load dataset
    X, y = load_dataset(root='../')

    # 2- drop non-necessary columns
    cols_names = ['id', 'dropoff_datetime']
    X, y = cols_to_drop(X, y, cols_names)
    print(f'X shape: {X.shape}, y shape: {y.shape}')

    # 3- get cat cols to encode and drop these cols from the dataframes
    # pass X or even y, chose any of them
    train_cat_cols, val_cat_cols, cat_cols_names, is_series = cat_cols_to_encode(X, y)
    X, y = cols_to_drop(X, y, cat_cols_names, is_series=is_series)
    print(f'X shape: {X.shape}, y shape: {y.shape}')

    # 4- convert pickup_datetime(object -str-) to datetime
    X_date, y_date = obj_to_datetime(X, y)
    cat_cols_names = 'pickup_datetime'
    X, y = cols_to_drop(X, y, cat_cols_names, is_series=True)
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    print(f'type of datetime col in numpy array: {type(X_date[0, 0])}')

    # 5- split datasets
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 6- convert (X_train, X_val, y_train and y_val) to numpy array
    X_train, X_val, y_train, y_val = datasets_to_numpy_array(X_train, X_val, y_train, y_val)
    train_cat_cols = cats_to_numpy_array(train_cat_cols)
    val_cat_cols = cats_to_numpy_array(val_cat_cols)

    # 7- encode cat feats using numpy array
    train_encoded_feats = encode_data(train_cat_cols, is_series)
    val_encoded_feats = encode_data(val_cat_cols, is_series)
    # print(encoded_feats)

    # print(f'shape of X_train: {X_train.shape}')
    # print(f'shape of train_encoded_feats: {train_encoded_feats.shape}')
    # print(f'shape of val_encoded_feats: {val_encoded_feats.shape}')
    # print(f'shape of X_date: {X_date.shape}')
    # 8- concatenate all numpy arrays, each to it's dataset
    # X_date = X_date.astype('object')
    X_train = np.concatenate([X_train, train_encoded_feats], axis=1)
    X_val = np.concatenate([X_val, val_encoded_feats], axis=1)

    # 9- scale
    X_train, X_val, y_train, y_val = scale_data(X_train, X_val, y_train, y_val)

    modeeel = lin_reg(X_train, y_train)

    r2 = eval_model(modeeel, X_train, y_train)
    print(r2)

    # r2 = eval_model(modeeel, X_val, y_val)
    # print(r2)

    print("second baseline model")


# if __name__ == '__main__':
    # 1- load dataset
    # X, y = load_dataset(root='../')
    #
    # # 2- drop non-necessary columns
    # cols_names = ['id', 'dropoff_datetime']
    # X, y = cols_to_drop(X, y, cols_names)
    # print(f'X shape: {X.shape}, y shape: {y.shape}')
    #
    # # 3- split cat & numerical columns
    # cat_col = get_cat_col(X)
    # features_names = cat_col.name
    # # print(cat_col.shape[1])
    # cat_col = cat_col.values
    # cat_col = cat_col.reshape(-1, 1)
    #
    # numeric_col = get_numeric_col(X)
    #
    # # 4- split dataset
    # X_train, X_val, y_train, y_val = split_data(X, y)
    #
    # # 5- encode data with (OHE)
    # transformed_train, categories_train = encode_data(X_train, cat_col)
    # transformed_val, categories_val = encode_data(X_val, cat_col)
    # print(f'X shape: {X.shape}, y shape: {y.shape}')
    # # print(f'OneHotEncoder encoded data: {encoded_data}')
    #
    # # 6- new DataFrame
    # X_train = concat_encoded_cols(X_train, features_names, transformed_train, categories_train)
    # X_val = concat_encoded_cols(X_val, features_names, transformed_val, categories_val)
    # print(X_train.shape)
    # print(f'DataFrame columns names: {list(X_train.columns)}')
    #
    # # 7- transform (X_train, X_val, y_train and y_val) to numpy array
    # X_train, X_val, y_train, y_val = to_numpy_array(X_train, X_val, y_train, y_val)
    # # print(X_train)
    #
    # # 8- scale data
    # X_train, X_val = scale_data(X_train, X_val)
    #
    # # lin_model = lin_reg(X_train, t_train)
    # #
    # # r2 = eval_model(lin_model, encoded_data, t_val)
    # # print(r2)
    #
    # print("baseline model")
