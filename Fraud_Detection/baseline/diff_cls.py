import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# solve problem cannot find module named Fraud_Detection
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from Fraud_Detection.data_helper import load_undersampled, load_original
from Fraud_Detection.data_helper import split_data, fold_original, grid_search


def select_classifiers():
    classifiers = {
        "LogisiticRegression": LogisticRegression(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    return classifiers


def lr_grid(X_train, y_train):
    cls = LogisticRegression(solver='liblinear')

    # LogisticRegression parameters for GridSearchCV
    param_grid = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10]}

    best_estimator, best_params = grid_search(classifier=cls, X_train=X_train, y_train=y_train, params=param_grid)
    print(best_params)

    return best_estimator


def knn_grid(X_train, y_train):
    knn_cls = KNeighborsClassifier()

    # KNeastNeighbors parameters for GridSearchCV
    param_grid = {'n_neighbors': list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    best_estimator, best_params = grid_search(classifier=knn_cls, X_train=X_train, y_train=y_train, params=param_grid)
    print(best_params)

    return best_estimator


def svc_grid(X_train, y_train):
    svc_cls = SVC()

    # SVC parameters for GridSearchCV
    param_grid = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    best_estimator, best_params = grid_search(classifier=svc_cls, X_train=X_train, y_train=y_train, params=param_grid)
    print(best_params)

    return best_estimator


def dt_grid(X_train, y_train):
    dt_cls = DecisionTreeClassifier()

    # DecisionTree parameters for GridSearchCV
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 4)),
                 'min_samples_leaf': list(range(5, 7))}

    best_estimator, best_params = grid_search(classifier=dt_cls, X_train=X_train, y_train=y_train, params=param_grid)
    print(best_params)

    return best_estimator


def rf_grid(X_train, y_train, **kwargs):
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


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score


if __name__ == '__main__':
    df = load_original()

    print('No Frauds', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the dataset')
    print('Frauds', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the dataset')

    X_original = df.drop('Class', axis=1)
    y_original = df['Class']

    original_X_train, original_X_test, original_y_train, original_y_test = fold_original(X_original, y_original)

    lr_grid = lr_grid(original_X_train, original_y_train)
    knn_grid = knn_grid(original_X_train, original_y_train)
    svc_grid = svc_grid(original_X_train, original_y_train)
    dt_grid = dt_grid(original_X_train, original_y_train)

    lr_pred = cross_val_predict(lr_grid, original_X_train, original_y_train, cv=5)
    knearst_pred = cross_val_predict(knn_grid, original_X_train, original_y_train, cv=5)
    svc_pred = cross_val_predict(svc_grid, original_X_train, original_y_train, cv=5)
    dt_pred = cross_val_predict(dt_grid, original_X_train, original_y_train, cv=5)

    # print(lr_pred)
    # np.unique(lr_pred)
    print(roc_auc_score(original_y_train, lr_pred))
    print(roc_auc_score(original_y_train, knearst_pred))
    print(roc_auc_score(original_y_train, svc_pred))
    print(roc_auc_score(original_y_train, dt_pred))

    # roc_auc(original_y_train, lr_pred, knearst_pred, svc_pred, dt_pred)
    lr_fpr, lr_tpr, lr_threshold = roc_curve(original_y_train, lr_pred)
    plt.plot([0, 1], [0, 1], 'b--')
    plt.plot(lr_fpr, lr_tpr)
    plt.show()


