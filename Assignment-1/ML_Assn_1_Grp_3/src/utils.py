# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008
# Assignment 1

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

class DecisionTree:     # Forward declaration
    pass

def entropy(labels):
    """
    Calculates the entropy of a set of values.

    Args:
        labels (pd.Series): The set of values for which the entropy has to be calculated.

    Returns:
        float: The obtained entropy.
    """
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [-1 * (x / total) * math.log2(x / total) for x in diff_vals]
    return sum(diff_vals)


def information_gain(data, labels, attr, split_val):
    """
    Calculates the information gain if we split the data on the basis of an attribute into 2 halves.
    One half with value < split_val, and the other with value >= split_val.

    Args:
        data (pd.DataFrame): The dataset we are working with, which needs to be split.

        labels (pd.Series): The 0, 1 outcomes corresponding to the training dataset.

        attr (str): The attribute for which we need to find the information gain.

        split_val (Union[int, float]): The value of the attribute around which we want to split.

    Returns:
        float: [description]
    """
    filt = data[attr] < split_val
    left_labels = labels[filt]
    right_labels = labels[~filt]
    gain = entropy(labels) - (len(left_labels) * entropy(left_labels) + len(right_labels) * entropy(right_labels))/len(labels)
    return gain


def gini_index(labels):
    """
    Calculates the gini index of a set of values.

    Args:
        labels (pd.Series): The set of values for which the gini index has to be calculated.

    Returns:
        float: The obtained gini index.
    """
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [(x / total) ** 2 for x in diff_vals]
    return 1.0 - sum(diff_vals)


def gini_gain(data, labels, attr, split_val):
    """
    Calculates the gini gain if we split the data on the basis of an attribute into 2 halves.
    One half with value < split_val, and the other with value >= split_val.

    Args:
        data (pd.DataFrame): The dataset we are working with, which needs to be split.

        labels (pd.Series): The 0, 1 outcomes corresponding to the training dataset.

        attr (str): The attribute for which we need to find the gini gain.

        split_val (Union[int, float]): The value of the attribute around which we want to split.

    Returns:
        float: [description]
    """
    gini = gini_index(labels)
    filt = data[attr] < split_val
    left_labels = labels[filt]
    right_labels = labels[~filt]
    new_gini = (len(left_labels) * gini_index(left_labels) + len(right_labels) * gini_index(right_labels)) / len(labels)
    return gini - new_gini


def find_best_split(data, labels, attr, measure):
    """
    Finds the best value of an attribute for splitting, given that particular attribute.
    Sort the values taken by the attribute in ascending order. Then for the midpoint of each
    pair of consecutive values, check the feasibility of that point for splitting.

    Args:
        data (pd.DataFrame): The dataset we are working with, which needs to be split.

        labels (pd.Series): The 0, 1 outcomes corresponding to the training dataset.

        attr (str): The attribute for which we need to find the gini gain.

        measure (str): The impurity measure to be used - information gain or gini gain.

    Returns:
        Tuple[Union[int, float], float]: [description]
    """
    # sort the values taken by the attribute and remove duplicates
    vals = np.sort(data[attr].unique())
    best_gain = 0
    best_split_val = None
    for i in range(len(vals) - 1):
        split_val = (vals[i] + vals[i + 1]) / 2         # get midpoint
        if measure == 'ig':
            gain = information_gain(data, labels, attr, split_val)
        else:
            gain = gini_gain(data, labels, attr, split_val)
        if best_gain < gain:
            best_gain = gain
            best_split_val = split_val

    return (best_split_val, best_gain)


def split_data(df, frac_1=0.8, frac_2=0.2):
    """
    Splits a given dataset into a random partition according to the given ratio.

    Args:
        df (pd.DataFrame): The dataset to be partitioned.

        frac_1 (float, optional): First fraction for partitioning the dataset. Defaults to 0.8.

        frac_2 (float, optional): Second fraction for partitioning the dataset. Defaults to 0.2.

    Raises:
        ValueError: An error is raised if the sum of the two fractions is not 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The two partitioned datasets.
    """
    if abs((frac_1 + frac_2) - 1.0) > 1e-6:
        raise ValueError('Fractions should add up to 1')
    
    df = df.sample(frac=1.0).reset_index(drop=True)     # shuffle the data
    ind = int(len(df.index) * frac_1)
    df_1 = df.iloc[:ind, :].reset_index(drop=True)      # get first partition
    df_2 = df.iloc[ind:, :].reset_index(drop=True)      # get second partition
    return (df_1, df_2)


def split_df_col(df):
    """
    Splits a dataframe into a dataframe having only the attribute columns, and a series
    having the outcome values.

    Args:
        df (pd.DataFrame): The dataframe that needs to be split.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The dataframe having all attribute columns, and
                the series having outcome labels.
    """
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return (data, labels)


def get_pred_accuracy(tree, test):
    """
    Retrieves the predictions and accuracy of the predictions for a given decision tree
    and test dataset.

    Args:
        tree (DecisionTree): The decision tree on the basis of which predictions 
                need to be made.

        test (pd.DataFrame): The test dataset.

    Returns:
        Tuple[pd.Series, float]: The outcomes as a series of 0, 1 values, and the 
                accuracy value.
    """
    test_data, test_labels = split_df_col(test)
    preds = tree.predict(test_data)
    accuracy = np.mean(preds == test_labels) * 100
    return (preds, accuracy)


def save_plot(x, y, param):
    """
    Creates a plot of y v/s x and saves it.

    Args:
        x (List): The values on the horizontal axis.

        y (List): The corresponding values on the vertical axis.

        param (str): A parameter describing the type of the plot, 'depth' for the accuracy v/s depth 
                plot, or 'nodes' for the accuracy v/s  no. of nodes plot.
    """
    label = ('Depth' if param == 'depth' else 'No. of Nodes')
    plt.figure()
    plt.title(f'Test Accuracy v/s {label}')
    plt.xlabel(label)
    plt.ylabel('Test Accuracy')
    if param == 'depth':
        plt.ylim(60, 80)
    plt.plot(x, y)
    plt.savefig(f'{param}_accuracy.png')


def precision(preds, labels):
    """
    Calculates the precision value based on predictions and true labels.
    Precision = True Positive / (True Positive + False Positive)

    Args:
        preds (pd.Series): The prediction values.

        labels (pd.Series): The true labels.

    Returns:
        float: The calculated precision value.
    """
    true_pos = np.sum(np.logical_and(labels, preds))
    return true_pos / (np.sum(preds) + 1e-6)            # the 1e-6 in the denominator prevents it from becoming 0


def recall(preds, labels):
    """
    Calculates the recall value based on predictions and true labels.
    Recall = True Positive / (True Positive + False Negative)

    Args:
        preds (pd.Series): The prediction values.

        labels (pd.Series): The true labels.

    Returns:
        float: The calculated recall value.
    """
    true_pos = np.sum(np.logical_and(labels, preds))
    return true_pos / (np.sum(labels) + 1e-6)           # the 1e-6 in the denominator prevents it from becoming 0


def f1_score(preds, labels):
    """
    Calculates the F1 score based on predictions and true labels.
    It is the harmonic mean of the precision and recall.

    Args:
        preds (pd.Series): The prediction values.

        labels (pd.Series): The true labels.

    Returns:
        float: The calculated F1 score.
    """
    p = precision(preds, labels)
    r = recall(preds, labels)
    return 2 * (p * r) / (p + r + 1e-6)         # the 1e-6 in the denominator prevents it from becoming 0
