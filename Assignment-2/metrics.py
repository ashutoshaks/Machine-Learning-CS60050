from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
import math

def display_metrics(pred_labels: np.ndarray, true_labels: np.ndarray, label_map: Dict[int, str]):
    pass


def get_metrics(
    pred_labels=None,
    true_labels=None,
    metrics: List[str] = ['Accuracy'],
    classes: List[str] = None,
) -> Dict:

    if isinstance(pred_labels, np.ndarray) == False:
        pred_labels = pred_labels.to_numpy()
    if isinstance(true_labels, np.ndarray) == False:
        true_labels = true_labels.to_numpy()

    results = {}

    for metric in metrics:

        if metric == 'Accuracy':
            results[metric] = accuracy(pred_labels, true_labels)

        elif metric == 'Precision':
            results[metric] = np.zeros(len(classes))
            for i, label in enumerate(classes):
                results[metric][i] = (precision_score(
                    pred_labels, true_labels, label=i))

        elif metric == 'Recall':
            results[metric] = np.zeros(len(classes))
            for i, label in enumerate(classes):
                results[metric][i] = (recall_score(
                    pred_labels, true_labels, label=i))

        elif metric == 'F1':
            results[metric] = np.zeros(len(classes))
            for i, label in enumerate(classes):
                results[metric][i] = (f1_score(
                    pred_labels, true_labels, label=i))

        elif metric == 'Confusion Matrix':
            results[metric] = confusion_matrix(
                pred_labels, true_labels, labels=classes)

        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    return results


def accuracy(pred_labels, true_labels):
    return np.mean(true_labels == pred_labels)


def precision_score(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / (np.sum(pred_labels == label) + 1e-7)


def recall_score(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / np.sum(true_labels == label)


def f1_score(pred_labels, true_labels, label: int):
    precision = precision_score(pred_labels, true_labels, label)
    recall = recall_score(pred_labels, true_labels, label)
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def confusion_matrix(pred_labels, true_labels, labels: List = []):
    matrix = np.zeros((len(labels), len(labels)))
    for label, label_name in enumerate(labels):
        true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
        matrix[label, label] = true_pos
        for i, other_label in enumerate(labels):
            if i != label:
                matrix[label, i] = np.sum(
                    np.logical_and(true_labels == label, pred_labels == i))
    return matrix


def find_ci_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float, float]:
    if isinstance(data, np.ndarray) == False:
        data = np.array(data)
    n = len(data)
    mean, std = data.mean(axis=0), data.std(axis=0)
    h = std * stats.t.ppf((1+confidence)/2., n-1)
    return mean, (mean-h, mean+h)


def ci(accuracy, n):
    d = 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)
    return (accuracy - d, accuracy + d)
