import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold



def load_original():
    data_path = '../data/trainval.csv'
    df = pd.read_csv(data_path)

    return df


def load_undersampled():
    data_path = 'Fraud_Detection/data/undersampled_data/data.csv'
    df = pd.read_csv(data_path)

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
    mm_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    X['scaled_amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['scaled_time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

    X.drop(['Time', 'Amount'], axis=1, inplace=True)

    # subX = X[['scaled_amount', 'scaled_time']]
    # X.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    #
    # X.insert(0, 'scaled_amount', subX['scaled_amount'])
    # X.insert(0, 'scaled_time', subX['scaled_time'])

    return X


def undersample_data(data):
    n_fraud = data['Class'].value_counts()[1]
    n_non_fraud = data['Class'].value_counts()[0]
    print(f'fraud {n_fraud}')
    # print(f'fraud {n_non_fraud}')
    n_rows = 305
    fraud = data.loc[data['Class'] == 1]
    non_fraud = data.loc[data['Class'] == 0][:n_rows]
    # print(fraud.shape)

    new_data = pd.concat([fraud, non_fraud], axis=0)

    # use pd.sample to shuffle the new DataFrame
    # Important step to prevent bias during training, ensure randomness in case of batch selection
    new_data = new_data.sample(frac=1)

    return new_data


def fold_original(X, y):
    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    # We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
    # original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the Distribution of the labels

    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    # See if both the train and test label distribution are similarly distributed
    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
    print('-' * 100)

    print('Label Distributions: \n')
    print(train_counts_label / len(original_ytrain))
    print(test_counts_label / len(original_ytest))

    return original_Xtrain, original_Xtest, original_ytrain, original_ytest


def grid_search(classifier, X_train, y_train, params, scoring='recall'):
    grid_model = GridSearchCV(classifier, param_grid=params, scoring=scoring)
    grid_model.fit(X_train, y_train)

    model_best_estims = grid_model.best_estimator_

    return model_best_estims


def classifier_cv_score(cls, X, y, cv=5):
    score = cross_val_score(cls, X, y, cv=cv)
    print(f'acc = {score}')
    # print(type(score))

    score_avg = score.sum() / len(score)
    print(f'acc avg = {score_avg}')
    # print(type(score_avg))

    return score_avg


def classifier_cv_predict(cls, X, y, cv=5):
    pred = cross_val_predict(estimator=cls, X=X, y=y, cv=cv)

    # print(f'cv predict {pred}')
    # print(f'cv predict {len(pred)}')

    return pred


def conf_matrix(y_actual, y_pred, cls_name):
    cm = confusion_matrix(y_actual, y_pred)
    plt.title(cls_name)
    labels = ['Non-Fraud', 'Fraud']
    cmap = ['YlGn', 'BuPu']

    sns.heatmap(cm, annot=True, fmt='', cbar=True, cmap=cmap[0],
                xticklabels=labels, yticklabels=labels)

    plt.tight_layout()

    plt.show()


def classifier_report(y_actual, y_pred, cls_name):
    target_names = ['Non-Fraud', 'Fraud']
    print(cls_name)
    print(classification_report(y_actual, y_pred, target_names=target_names))


def test_original_data(cls, X, y):
    pred = classifier_cv_predict(cls, X, y)

    classifier_report(y, pred, 'Logistic Regression')
    conf_matrix(y, pred, 'Logistic Regression')


