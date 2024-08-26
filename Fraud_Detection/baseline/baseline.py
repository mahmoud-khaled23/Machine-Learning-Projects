"""
baseline model for Fraud Detection
"""

# solve problem cannot find module named Fraud_Detection
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from Fraud_Detection.data_helper import load_train, load_val
from Fraud_Detection.data_helper import make_undersample_data, split_data, preprocess_data
from Fraud_Detection.tester import classifier_cv_score, classifier_cv_predict
from Fraud_Detection.tester import test_val_data, get_roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def grid_search(classifier, X_train, y_train, params, **kwargs):
    grid_model = GridSearchCV(classifier, param_grid=params, **kwargs)
    grid_model.fit(X_train, y_train)

    best_estimator = grid_model.best_estimator_
    best_params = grid_model.best_params_

    return best_estimator, best_params


def random_forest(X_train, y_train, **kwargs):
    rf_cls = RandomForestClassifier()

    param_grid = {
        'n_estimators': [100, 200, 300, 400, 800],
        'max_depth': [3, 5, 7, 10],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2, 4]
    }

    best_estimator, best_params = grid_search(classifier=rf_cls, X_train=X_train, y_train=y_train,
                                              params=param_grid, **kwargs)
    print(f'Best Params: {best_params}')

    return best_estimator


if __name__ == '__main__':
    print(f'{127803:^20c}')
    print("I miss CODING :(")

    data = load_train()
    u_data = make_undersample_data(data)

    X = u_data.drop('Class', axis=1)
    y = u_data['Class']

    X = preprocess_data(X)

    rf_grid_cv = random_forest(X, y)

    classifier_cv_score(rf_grid_cv, X, y)

    X_val, y_val = load_val()

    test_val_data(rf_grid_cv, X_val, y_val)

    y_original_pred = classifier_cv_predict(rf_grid_cv, X_val, y_val)
    get_roc_auc_score(y_val, y_original_pred)

    print(f'Fraud Detection works pretty well')

