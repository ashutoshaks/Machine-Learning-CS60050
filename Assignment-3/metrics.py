# Machine Learning - Assignment 3
# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008

import numpy as np
from tabulate import tabulate
import math

def accuracy(pred_labels, true_labels):
    """
    Calculates the accuracy given the predictions and true labels.
    Accuracy = No. of instances correctly classified / Total no. of instances

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated accuracy value.
    """
    return np.mean(true_labels == pred_labels)


def precision(pred_labels, true_labels):
    """
    Calculates the precision (positive predictive value) value based on predictions and true labels.
    Precision = True Positive / (True Positive + False Positive)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated precision value.
    """
    true_pos = np.sum(np.logical_and(true_labels == 1, pred_labels == 1))
    return true_pos / (np.sum(pred_labels == 1) + 1e-10)


def recall(pred_labels, true_labels):
    """
    Calculates the sensitivity (recall) value based on predictions and true labels.
    Sensitivity = True Positive / (True Positive + False Negative)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated recall value.
    """
    true_pos = np.sum(np.logical_and(true_labels == 1, pred_labels == 1))
    return true_pos / np.sum(true_labels == 1)


def specificity(pred_labels, true_labels):
    """
    Calculates the specificity value based on predictions and true labels.
    Specificity = True Negative / (False Positive + True Negative)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated specificity value.
    """
    true_neg = np.sum(np.logical_and(true_labels != 1, pred_labels != 1))
    return true_neg / np.sum(true_labels != 1)


def negative_predictive_value(pred_labels, true_labels):
    """
    Calculates the negative predicitve value value based on predictions and true labels.
    Negative predicitve value = True Negative / (True Negative + False Negative)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated negative predicitve value value.
    """
    true_neg = np.sum(np.logical_and(true_labels != 1, pred_labels != 1))
    return true_neg / (np.sum(pred_labels != 1) + 1e-10)


def f1_score(pred_labels, true_labels):
    """
    Calculates the F1-score based on predictions and true labels.
    It is the harmonic mean of the precision and recall.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated F1-score.
    """
    precision_score = precision(pred_labels, true_labels)
    recall_score = recall(pred_labels, true_labels)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-10)


def prevalence(pred_labels, true_labels):
    """
    Calculates the prevalence value based on predictions and true labels.
    Prevalence is the proportion of all positives in the total number of observations.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated prevalence value.
    """
    return np.sum(true_labels == 1) / len(true_labels)


def detection_rate(pred_labels, true_labels):
    """
    Calculates the detection rate value based on predictions and true labels.
    Detection rate is the proportion of true positives in the total number of observations.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated detection rate value.
    """
    true_pos = np.sum(np.logical_and(true_labels == 1, pred_labels == 1))
    return true_pos / len(true_labels)


def detection_prevalence(pred_labels, true_labels):
    """
    Calculates the detection prevalence value based on predictions and true labels.
    Detection prevalence is the number of positive class predictions made as a proportion of all predictions.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated detection prevalence value.
    """
    return np.sum(pred_labels == 1) / len(pred_labels)


def balanced_accuracy(pred_labels, true_labels):
    """
    Calculates the balanced accuracy value based on predictions and true labels.
    Balanced accuracy is the average of the sensitivity and the specificity.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

    Returns:
        float: The calculated balanced accuracy value.
    """
    sensitivity_score = recall(pred_labels, true_labels)
    specificity_score = specificity(pred_labels, true_labels)
    return (sensitivity_score + specificity_score) / 2


def confusion_matrix(pred_labels, true_labels):
    """
    Calculates and returns the confusion matrix.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label_map (Dict): A dictionary mapping labels to author names.

    Returns:
        np.ndarray: The confusion matrix.
    """
    matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            matrix[i, j] = np.sum(np.logical_and(true_labels == i, pred_labels == j))
    return matrix


def ci_95(accuracy, num_instances):
    """
    Calculates the 95% confidence interval of accuracy.

    Args:
        accuracy (float): The accuracy value.
        num_instances (int): The number of instances.

    Returns:
        Tuple[float, float]: The lower and upper limits of the 95% confidence interval.
    """
    d = 1.960 * math.sqrt(accuracy * (1 - accuracy) / num_instances)
    return (accuracy - d, accuracy + d)


def display_metrics(pred_labels, true_labels):
    """
    Displays all the required metrics in a formatted manner.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label_map (Dict): A dictionary mapping labels to author names.
    """
    headers = [1, 0]
    headers.insert(0, '')
    statistics = dict()
    
    statistics['Precision (Positive Predictive Value)'] = precision(pred_labels, true_labels)
    statistics['Recall (Sensitivity)'] = recall(pred_labels, true_labels)
    statistics['Specificity'] = specificity(pred_labels, true_labels)
    statistics['Negative Predictive Value'] = negative_predictive_value(pred_labels, true_labels)
    statistics['F1-Score'] = f1_score(pred_labels, true_labels)
    statistics['Prevalence'] = prevalence(pred_labels, true_labels)
    statistics['Detection Rate'] = detection_rate(pred_labels, true_labels)
    statistics['Detection Prevalence'] = detection_prevalence(pred_labels, true_labels)
    statistics['Balanced Accuracy'] = balanced_accuracy(pred_labels, true_labels)

    stats_list = []
    for (key, value) in statistics.items():
        stats_list.append([key, value])

    matrix = confusion_matrix(pred_labels, true_labels)
    matrix_print = []
    for label in [0, 1]:
        matrix_print.append([1 - label, *matrix[label]])
    print('\nConfusion Matrix:\n')
    print(tabulate(matrix_print, headers=headers, tablefmt='fancy_grid'))

    print('\nStatistics:\n')
    print(tabulate(stats_list, floatfmt=".4f", tablefmt='fancy_grid'))

    print('\nOverall Statistics:\n')
    acc = accuracy(pred_labels, true_labels)
    ci = ci_95(acc, len(true_labels))
    
    overall_stats = [['Accuracy', acc * 100], ['95% Confidence Interval', (ci[0] * 100, ci[1] * 100)]]
    print(tabulate(overall_stats, floatfmt=".4f", tablefmt='fancy_grid'))
