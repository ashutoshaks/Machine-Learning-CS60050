# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008
# Machine Learning - Assignment 2

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


def precision(pred_labels, true_labels, label):
    """
    Calculates the precision (positive predictive value) value based on predictions and true labels.
    Precision = True Positive / (True Positive + False Positive)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which precision has to be calculated.

    Returns:
        float: The calculated precision value.
    """
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / (np.sum(pred_labels == label) + 1e-10)


def sensitivity(pred_labels, true_labels, label):
    """
    Calculates the sensitivity (recall) value based on predictions and true labels.
    Sensitivity = True Positive / (True Positive + False Negative)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which sensitivity has to be calculated.

    Returns:
        float: The calculated sensitivity value.
    """
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / np.sum(true_labels == label)


def specificity(pred_labels, true_labels, label):
    """
    Calculates the specificity value based on predictions and true labels.
    Specificity = True Negative / (False Positive + True Negative)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which specificity has to be calculated.

    Returns:
        float: The calculated specificity value.
    """
    true_neg = np.sum(np.logical_and(true_labels != label, pred_labels != label))
    return true_neg / np.sum(true_labels != label)


def negative_predictive_value(pred_labels, true_labels, label):
    """
    Calculates the negative predicitve value value based on predictions and true labels.
    Negative predicitve value = True Negative / (True Negative + False Negative)

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which negative predicitve value has to be calculated.

    Returns:
        float: The calculated negative predicitve value value.
    """
    true_neg = np.sum(np.logical_and(true_labels != label, pred_labels != label))
    return true_neg / (np.sum(pred_labels != label) + 1e-10)


def f1_score(pred_labels, true_labels, label):
    """
    Calculates the F1-score based on predictions and true labels.
    It is the harmonic mean of the precision and recall.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which F1-score has to be calculated.

    Returns:
        float: The calculated F1-score.
    """
    precision_score = precision(pred_labels, true_labels, label)
    recall_score = sensitivity(pred_labels, true_labels, label)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-10)


def prevalence(pred_labels, true_labels, label):
    """
    Calculates the prevalence value based on predictions and true labels.
    Prevalence is the proportion of all positives in the total number of observations.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which prevalence has to be calculated.

    Returns:
        float: The calculated prevalence value.
    """
    return np.sum(true_labels == label) / len(true_labels)


def detection_rate(pred_labels, true_labels, label):
    """
    Calculates the detection rate value based on predictions and true labels.
    Detection rate is the proportion of true positives in the total number of observations.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which detection rate has to be calculated.

    Returns:
        float: The calculated detection rate value.
    """
    true_pos = np.sum(np.logical_and(true_labels == label, pred_labels == label))
    return true_pos / len(true_labels)


def detection_prevalence(pred_labels, true_labels, label):
    """
    Calculates the detection prevalence value based on predictions and true labels.
    Detection prevalence is the number of positive class predictions made as a proportion of all predictions.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which detection prevalence has to be calculated.

    Returns:
        float: The calculated detection prevalence value.
    """
    return np.sum(pred_labels == label) / len(pred_labels)


def balanced_accuracy(pred_labels, true_labels, label):
    """
    Calculates the balanced accuracy value based on predictions and true labels.
    Balanced accuracy is the average of the sensitivity and the specificity.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label (int): The class label for which balanced accuracy has to be calculated.

    Returns:
        float: The calculated balanced accuracy value.
    """
    sensitivity_score = sensitivity(pred_labels, true_labels, label)
    specificity_score = specificity(pred_labels, true_labels, label)
    return (sensitivity_score + specificity_score) / 2


def confusion_matrix(pred_labels, true_labels, label_map):
    """
    Calculates and returns the confusion matrix.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label_map (Dict): A dictionary mapping labels to author names.

    Returns:
        np.ndarray: The confusion matrix.
    """
    matrix = np.zeros((len(label_map), len(label_map)))
    for i in range(len(label_map)):
        for j in range(len(label_map)):
            matrix[i, j] = np.sum(np.logical_and(pred_labels == i, true_labels == j))
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


def display_metrics(pred_labels, true_labels, label_map):
    """
    Displays all the required metrics in a formatted manner.

    Args:
        pred_labels (np.ndarray): The predictions from our model.

        true_labels (np.ndarray): The true labels for the test data.

        label_map (Dict): A dictionary mapping labels to author names.
    """
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
    print(tabulate(matrix_print, headers=headers, tablefmt='fancy_grid'))

    print('\nStatistics by Class:\n')
    print(tabulate(stats_list, headers=headers, floatfmt=".4f", tablefmt='fancy_grid'))

    precision_avg = np.sum(statistics['Precision (Positive Predictive Value)']) / len(statistics['Precision (Positive Predictive Value)'])
    recall_avg = np.sum(statistics['Sensitivity (Recall)']) / len(statistics['Sensitivity (Recall)'])
    f1_macro_avg = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg)

    class_avg_stats = [['Average Precision', precision_avg], ['Average Recall (Sensitivity)', recall_avg], ['Macro-Averaged F1-Score', f1_macro_avg]]
    print(tabulate(class_avg_stats, floatfmt=".4f", tablefmt='fancy_grid'))

    print('\nOverall Statistics:\n')
    acc = accuracy(pred_labels, true_labels)
    ci = ci_95(acc, len(true_labels))
    
    overall_stats = [['Accuracy', acc * 100], ['95% Confidence Interval', (ci[0] * 100, ci[1] * 100)]]
    print(tabulate(overall_stats, floatfmt=".4f", tablefmt='fancy_grid'))
