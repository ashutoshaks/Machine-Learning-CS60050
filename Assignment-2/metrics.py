from typing import List, Tuple, Dict
import numpy as np
from tabulate import tabulate
import math


def accuracy(pred_labels, true_labels):
    return np.mean(true_labels == pred_labels)


def precision(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / (np.sum(pred_labels == label) + 1e-10)


def sensitivity(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / np.sum(true_labels == label)


def specificity(pred_labels, true_labels, label: int):
    true_neg = np.sum(np.logical_and(true_labels != label, pred_labels != label))
    return true_neg / np.sum(true_labels != label)


def negative_predictive_value(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels != label, pred_labels != label))
    return true_pos / (np.sum(pred_labels != label) + 1e-10)


def f1_score(pred_labels, true_labels, label: int):
    precision_score = precision(pred_labels, true_labels, label)
    recall_score = sensitivity(pred_labels, true_labels, label)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-10)


def prevalence(pred_labels, true_labels, label: int):
    return np.sum(true_labels == label) / len(true_labels)


def detection_rate(pred_labels, true_labels, label: int):
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / len(true_labels)


def detection_prevalence(pred_labels, true_labels, label: int):
    return np.sum(pred_labels == label) / len(pred_labels)


def balanced_accuracy(pred_labels, true_labels, label: int):
    sensitivity_score = sensitivity(pred_labels, true_labels, label)
    specificity_score = specificity(pred_labels, true_labels, label)
    return (sensitivity_score + specificity_score) / 2


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
    
    statistics['Precision (Positive Predictive Value)'] = [precision(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Sensitivity (Recall)'] = [sensitivity(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Specificity'] = [specificity(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Negative Predictive Value'] = [negative_predictive_value(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['F1-Score'] = [f1_score(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Prevalence'] = [prevalence(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Detection Rate'] = [detection_rate(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Detection Prevalence'] = [detection_prevalence(pred_labels, true_labels, i) for i in range(len(label_map))]
    statistics['Balanced Accuracy'] = [balanced_accuracy(pred_labels, true_labels, i) for i in range(len(label_map))]

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

    precision_avg = np.sum(statistics['Precision (Positive Predictive Value)']) / len(statistics['Precision (Positive Predictive Value)'])
    recall_avg = np.sum(statistics['Sensitivity (Recall)']) / len(statistics['Sensitivity (Recall)'])
    f1_macro_avg = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg)

    print(f'\nAverage Precision: {precision_avg:.4f}')
    print(f'Average Recall (Sensitivity): {recall_avg:.4f}')
    print(f'Macro-Averaged F1-Score: {f1_macro_avg:.4f}')

    print('\nOverall Statistics:\n')
    acc = (accuracy(pred_labels, true_labels))
    ci = ci_95(acc, len(true_labels))
    nir = np.max(np.bincount(true_labels)) / len(true_labels)
    print(f'Accuracy: {(acc * 100):.4f}%')
    print(f'95% Confidence Interval: ({(ci[0] * 100):.4f}%, {(ci[1] * 100):.4f}%)')
    print(f'No Information Rate: {nir:.4f}')
