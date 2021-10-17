from os import stat
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import math


def accuracy(pred_labels, true_labels):
    return np.mean(true_labels == pred_labels)


def precision(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / (np.sum(pred_labels == label) + 1e-8)


def sensitivity(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / np.sum(true_labels == label)

def specificity(pred_labels, true_labels, label: int):
    true_neg = np.sum(np.logical_and(true_labels != label, pred_labels != label))
    return true_neg / np.sum(true_labels != label)


def f1_score(pred_labels, true_labels, label: int):
    p = precision(pred_labels, true_labels, label)
    r = sensitivity(pred_labels, true_labels, label)
    return 2 * (p * r) / (p + r + 1e-8)


def confusion_matrix(pred_labels, true_labels, label_map: Dict):
    matrix = np.zeros((len(label_map), len(label_map)))
    for i in range(len(label_map)):
        for j in range(len(label_map)):
            matrix[i, j] = np.sum(np.logical_and(pred_labels == i, true_labels == j))
    return matrix


def ci_95(accuracy, n):
    d = 1.960 * math.sqrt(accuracy * (1 - accuracy) / n)
    return (accuracy - d, accuracy + d)


def display_metrics(pred_labels: np.ndarray, true_labels: np.ndarray, label_map: Dict[int, str]):
    headers = [name for name in label_map.values()]
    headers.insert(0, '')
    statistics = dict()
    
    statistics['Precision'] = [precision(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Sensitivity'] = [sensitivity(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Specificity'] = [specificity(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['F1-Score'] = [f1_score(pred_labels, true_labels, i) for i in range(len(label_map))]

    stats_list = [];
    for (key, value) in statistics.items():
        stats_list.append([key, *value])

    matrix = confusion_matrix(pred_labels, true_labels, label_map)
    matrix_print = []
    for (key, value) in label_map.items():
        matrix_print.append([value, *matrix[key]])
    print('Confusion Matrix:\n')
    print(tabulate(matrix_print, headers=headers, tablefmt='presto'))

    print('\nStatistics by Class:\n')
    print(tabulate(stats_list, headers=headers, floatfmt=".4f", tablefmt='presto'))

    print('\nOverall Statistics:\n')
    acc = (accuracy(pred_labels, true_labels))
    ci = ci_95(acc, len(true_labels))
    print(f'Accuracy: {(acc * 100):.4f}%')
    print(f'95% Confidence Interval: ({(ci[0] * 100):.4f}%, {(ci[1] * 100):.4f}%)')
