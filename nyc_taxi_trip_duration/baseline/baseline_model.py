from data_manage import *
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


def lin_reg(X_train, t_train):
    model = LinearRegression()
    model = Ridge(alpha=1, fit_intercept=True)
    model.fit(X_train, t_train)

    return model


def eval_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return r2


if __name__ == '__main__':
    with open("../conf/baseline_config.yml", 'r') as file:
        config = yaml.safe_load(file)

    # 1- load dataset
    train, test = load_dataset(root='../')

    # 2- drop non-necessary columns
    train, test = cols_to_drop(train, test, config["drop_cols"])
    print(f'X shape: {train.shape}, y shape: {test.shape}')

    X_train, y_train, X_test, y_test = split_data(train, test)
    print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

    X_train, X_test = encode_(X_train, X_test, config["encode_cols"])
    print(f'enc_train shape: {X_train.shape}, enc_test shape: {X_test.shape}')
    print(f'enc_train shape: {type(X_train)}, enc_test shape: {X_test.shape}')

    # 5- split datasets
    X_train, X_test = extract_datetime(X_train, X_test, config["datetime_cols"])

    # print(X_train[0:5, -1])
    print(f'after extract')
    print(f'columns names: {X_train.columns}')
    print(X_train.head(5))

    X_train, y_train, X_test, y_test = datasets_to_numpy_array(X_train, y_train, X_test, y_test)

    print(X_train[0:5, -1])

    # print(encoded_feats)
    # 9- scale
    X_train, X_test = scale_data(X_train, X_test, processor=StandardScaler())

    model = lin_reg(X_train, y_train)

    r2 = eval_model(model, X_test, y_test)
    print(r2)
    #
    # r2 = eval_model(modeeel, X_val, y_val)
    # print(r2)

    print("second baseline model")
