from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
import math


def get_metrics(
    y_pred=None,
    y_true=None,
    metrics: List[str] = ['Accuracy'],
    classes: List[str] = None,
) -> Dict:

    if isinstance(y_pred, np.ndarray) == False:
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, np.ndarray) == False:
        y_true = y_true.to_numpy()

    results = {}

    for metric in metrics:

        if metric == 'Accuracy':
            results[metric] = accuracy(y_true, y_pred)

        elif metric == 'Precision':
            results[metric] = np.zeros(len(classes))
            for i, label in enumerate(classes):
                results[metric][i] = (precision_score(
                    y_true, y_pred, label=i))

        elif metric == 'Recall':
            results[metric] = np.zeros(len(classes))
            for i, label in enumerate(classes):
                results[metric][i] = (recall_score(
                    y_true, y_pred, label=i))

        elif metric == 'F1':
            results[metric] = np.zeros(len(classes))
            for i, label in enumerate(classes):
                results[metric][i] = (f1_score(
                    y_true, y_pred, label=i))

        elif metric == 'Confusion Matrix':
            results[metric] = confusion_matrix(
                y_true, y_pred, labels=classes)

        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    return results


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, label: int):
    """
    Computes the precision score for a given set of labels.
    Precision score = true_pos / (true_pos + false_pos)
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label: The label to consider.
    Returns:
        Precision Score.
    """
    true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
    return true_pos / (np.sum(y_pred == label) + 1e-7)


def recall_score(y_true, y_pred, label: int):
    """
    Computes the recall score for a given set of labels.
    Recall score = true_pos / (true_pos + false_neg)
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label: The label to consider.
    Returns:
        Recall Score.
    """
    true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
    return true_pos / np.sum(y_true == label)


def f1_score(y_true, y_pred, label: int):
    """
    Computes the F1 score for a given set of labels.
    F1 score = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label: The label to consider.
    Returns:
        F1 score.
    """
    precision = precision_score(y_true, y_pred, label)
    recall = recall_score(y_true, y_pred, label)
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def confusion_matrix(y_true, y_pred, labels: List = []):
    """
    Computes the confusion matrix for a given set of labels.
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        labels: The list of labels to consider.
    Returns:
        The confusion matrix. (np.ndarray)
    """
    matrix = np.zeros((len(labels), len(labels)))
    for label, label_name in enumerate(labels):
        true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
        matrix[label, label] = true_pos
        for i, other_label in enumerate(labels):
            if i != label:
                matrix[label, i] = np.sum(
                    np.logical_and(y_true == label, y_pred == i))
    return matrix


def find_ci_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float, float]:
    """
    Finds the confidence interval for a given dataset of a estimator.
    """
    if isinstance(data, np.ndarray) == False:
        data = np.array(data)
    n = len(data)
    mean, std = data.mean(axis=0), data.std(axis=0)
    h = std * stats.t.ppf((1+confidence)/2., n-1)
    return mean, (mean-h, mean+h)

def ci(accuracy, n):
    d = 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)
    return (accuracy - d, accuracy + d)
