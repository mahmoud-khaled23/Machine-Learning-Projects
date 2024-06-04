import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from Fraud_Detection.data_helper import load_undersampled, load_original
from Fraud_Detection.data_helper import split_data, fold_original


def select_classifiers():
    classifiers = {
        "LogisiticRegression": LogisticRegression(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    return classifiers


def lr_params(X_train, y_train):
    # use GridSearchCV to find the best parameters
    from sklearn.model_selection import GridSearchCV

    # LogisticRegression parameters for GridSearchCV
    lr_params = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10]}

    lr_cv = GridSearchCV(LogisticRegression(solver='liblinear'), lr_params)
    lr_cv.fit(X_train, y_train)

    # LogisticRegression best estimators
    lr_estims = lr_cv.best_estimator_
    print(lr_cv.best_params_)

    return lr_estims


def knn_params(X_train, y_train):
    # KNeastNeighbors parameters for GridSearchCV
    knearst_params = {'n_neighbors': list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    knearst_cv = GridSearchCV(KNeighborsClassifier(), knearst_params)
    knearst_cv.fit(X_train, y_train)

    # KNearstNeighbors best estimators
    knearst_estims = knearst_cv.best_estimator_
    print(knearst_cv.best_params_)

    return knearst_estims


def svc_params(X_train, y_train):
    # SVC parameters for GridSearchCV
    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    svc_cv = GridSearchCV(SVC(), svc_params)
    svc_cv.fit(X_train, y_train)

    # SVC best estimator
    svc_estims = svc_cv.best_estimator_
    print(svc_cv.best_params_)

    return svc_estims


def dt_params(X_train, y_train):
    # DecisionTree parameters for GridSearchCV
    dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 4)),
                 'min_samples_leaf': list(range(5, 7))}

    dt_cv = GridSearchCV(DecisionTreeClassifier(), dt_params)
    dt_cv.fit(X_train, y_train)

    # DecisionTree best esimators
    dt_estims = dt_cv.best_estimator_
    print(dt_cv.best_params_)

    return dt_estims


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score


def roc_auc(original_y_train, lr_pred, knearst_pred, svc_pred, dt_pred):
    lr_fpr, lr_tpr, lr_threshold = roc_curve(original_y_train, lr_pred)
    knearst_fpr, knearst_tpr, knearst_threshold = roc_curve(original_y_train, knearst_pred)
    svc_fpr, svc_tpr, svc_threshold = roc_curve(original_y_train, svc_pred)
    dt_fpr, dt_tpr, dt_threshold = roc_curve(original_y_train, dt_pred)

    plt.plot([0, 1], [0, 1], 'b--')
    plt.plot(lr_fpr, lr_tpr)
    plt.plot(knearst_fpr, knearst_tpr)
    plt.plot(svc_fpr, svc_tpr)
    plt.plot(dt_fpr, dt_tpr)

    print(lr_threshold)


if __name__ == '__main__':
    df = load_original()

    print('No Frauds', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the dataset')
    print('Frauds', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the dataset')

    X_original = df.drop('Class', axis=1)
    y_original = df['Class']

    original_X_train, original_X_test, original_y_train, original_y_test = fold_original(X_original, y_original)

    lr_estims = lr_params(original_X_train, original_y_train)
    knearst_estims = knn_params(original_X_train, original_y_train)
    svc_estims = svc_params(original_X_train, original_y_train)
    dt_estims = dt_params(original_X_train, original_y_train)

    lr_pred = cross_val_predict(lr_estims, original_X_train, original_y_train, cv=5)
    knearst_pred = cross_val_predict(knearst_estims, original_X_train, original_y_train, cv=5)
    svc_pred = cross_val_predict(svc_estims, original_X_train, original_y_train, cv=5)
    dt_pred = cross_val_predict(dt_estims, original_X_train, original_y_train, cv=5)

    # print(lr_pred)
    # np.unique(lr_pred)
    # print(roc_auc_score(original_y_train, lr_pred))
    # print(roc_auc_score(original_y_train, knearst_pred))
    # print(roc_auc_score(original_y_train, svc_pred))
    # print(roc_auc_score(original_y_train, dt_pred))

    roc_auc(original_y_train, lr_pred, knearst_pred, svc_pred, dt_pred)

