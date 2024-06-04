"""
baseline model for Fraud Detection
"""
import numpy as np
import pandas as pd
import seaborn as sns
from Fraud_Detection.data_helper import undersample_data, split_data, grid_search, load_undersampled, preprocess_data
from Fraud_Detection.data_helper import classifier_cv_score, classifier_cv_predict
from Fraud_Detection.data_helper import conf_matrix, classifier_report
from Fraud_Detection.data_helper import test_original_data
from sklearn.linear_model import LogisticRegression

import imblearn


def logistic_regression(X_train, y_train):
    """
    apply Logistic Regression with
    :param X_train:
    :param y_train:
    :return:
    """
    classifier = LogisticRegression(solver='liblinear')
    params = {'penalty': ['l1', 'l2'],
              'C': [.001, .01, .1, 1, 10, 100]}
    scoring = 'recall'

    classifier_cv = grid_search(classifier, X_train, y_train, params, scoring=scoring)

    return classifier_cv


if __name__ == '__main__':
    print(f'{127803:^20c}')
    print("I miss CODING :(")
    x = 0
    # data = pd.DataFrame(pd.read_csv("../data/train.csv"))

    data = load_undersampled()
    # new_data = undersample_data(data)
    X = data.drop('Class', axis=1)
    y = data['Class']

    X = preprocess_data(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    lr_estims = logistic_regression(X_train, y_train)

    classifier_cv_score(lr_estims, X_train, y_train)

    # y_pred = classifier_cv_predict(lr_estims, X_train, y_train)
    y_pred = classifier_cv_predict(lr_estims, X_test, y_test)

    # classifier_report(y_test, y_pred, 'Logistic Regression')
    # conf_matrix(y_test, y_pred, 'Logistic Regression')

    X_original, y_original = data.drop('Class', axis=1), data['Class']
    test_original_data(lr_estims, X_original, y_original)

    print(f'Fraud Detection works pretty well')

