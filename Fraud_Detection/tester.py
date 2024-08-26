
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


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


def test_val_data(cls, X, y):
    pred = classifier_cv_predict(cls, X, y)

    rf_name = 'Random Forest Classifier'
    classifier_report(y, pred, rf_name)
    conf_matrix(y, pred, rf_name)


def roc_auc_plot(original_y_train, y_pred):
    lr_fpr, lr_tpr, lr_threshold = roc_curve(original_y_train, y_pred)

    plt.plot([0, 1], [0, 1], 'b--')
    plt.plot(lr_fpr, lr_tpr)
    plt.show()

    print(lr_threshold)


def get_roc_auc_score(original_y_train, y_pred):
    score = roc_auc_score(original_y_train, y_pred, average='macro')

    print(score)


def classifier_cv_score(cls, X, y, cv=5):
    score = cross_val_score(cls, X, y, cv=cv)
    print(f'acc = {score}')

    score_avg = score.sum() / len(score)
    print(f'acc avg = {score_avg}')

    return score_avg


def classifier_cv_predict(cls, X, y, cv=5):
    pred = cross_val_predict(estimator=cls, X=X, y=y, cv=cv)

    return pred


