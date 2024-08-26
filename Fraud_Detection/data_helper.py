#!/usr/bin/python
# use this line of code to let the kernel knows what interpreter to use

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_train():
    # data_path = '../data/train.csv'
    full_path = 'Fraud_Detection/data/train.csv'
    df = pd.read_csv(full_path)

    return df


def load_val():
    # data_path = '../data/train.csv'
    full_path = 'Fraud_Detection/data/val.csv'
    df = pd.read_csv(full_path)

    X, y = df.drop('Class', axis=1), df['Class']

    return X, y


def load_undersampled():
    # data_path = '../data/undersampled_data/data.csv'
    full_path = 'Fraud_Detection/data/undersampled_data/data.csv'
    df = pd.read_csv(full_path)

    return df


def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=37)

    # convert data to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    # print(type(y_test))

    return X_train, X_test, y_train, y_test


def preprocess_data(X, scaler=RobustScaler()):
    X['scaled_amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['scaled_time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

    X.drop(['Time', 'Amount'], axis=1, inplace=True)

    return X


def make_undersample_data(data):
    # data = load_train()
    data = data.sample(frac=1)

    n_rows = data['Class'].value_counts()[1]
    print(f'number of rows: {n_rows}')

    fraud = data.loc[data['Class'] == 1][:n_rows]
    non_fraud = data.loc[data['Class'] == 0][:n_rows]
    # print(fraud.shape)

    new_data = pd.concat([fraud, non_fraud], axis=0)

    # use pd.sample to shuffle the new DataFrame
    # Important step to prevent bias during training, ensure randomness in case of batch selection
    new_data = new_data.sample(frac=1)

    return new_data

