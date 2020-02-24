import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
import pandas as pd


def get_metrics(true_labels, predicted_labels):
    accuracy = np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        4)
    prec = np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        4)
    recall = np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        4)
    f1 = np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        4)

    df = pd.DataFrame([[accuracy, prec, recall, f1]], index=['performance'],
                      columns=["accuracy", "precision", "recall", "f1_score"])
    print(df)
    return df

def display_classification_report(true_labels, predicted_labels, classes=[1, 0]):
    report = metrics.classification_report(y_true=true_labels,
                                           y_pred=predicted_labels,
                                           labels=classes)
    return report


def display_confusion_matrix(true_labels, predicted_labels, classes=[1, 0]):
    total_classes = len(classes)
    level_labels = [total_classes * [0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels,
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm,
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes],
                                                  labels=level_labels),
                            index=pd.MultiIndex(levels=[['Actual:'], classes],
                                                labels=level_labels))
    return cm_frame


def display_model_performance_metrics(true_labels, predicted_labels, classes=[1, 0]):
    print('Model Performance metrics:')
    print('-' * 30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-' * 30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels,
                                  classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-' * 30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels,
                             classes=classes)


def predict_labels(model, X, y):
    y_pred = model.predict(X)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    y_pred = pd.Series(y_pred.ravel(), index=y.index)
    return y_pred
