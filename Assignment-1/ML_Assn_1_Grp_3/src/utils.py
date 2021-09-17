import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Union
from matplotlib import pyplot as plt

class DecisionTree:
    pass


def entropy(labels: pd.Series) -> float:
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [-1 * (x / total) * math.log2(x / total) for x in diff_vals]
    return sum(diff_vals)


def information_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    filt = data[attr] < split_val
    left_labels = labels[filt]
    right_labels = labels[~filt]
    gain = entropy(labels) - (len(left_labels) * entropy(left_labels) + len(right_labels) * entropy(right_labels))/len(labels)
    return gain


def gini_index(labels: pd.Series) -> float:
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [(x / total) ** 2 for x in diff_vals]
    return 1.0 - sum(diff_vals)


def gini_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    gini = gini_index(labels)
    filt = data[attr] < split_val
    left_labels = labels[filt]
    right_labels = labels[~filt]
    new_gini = (len(left_labels) * gini_index(left_labels) + len(right_labels) * gini_index(right_labels)) / len(labels)
    return gini - new_gini


def find_best_split(data: pd.DataFrame, labels: pd.Series, attr: str, measure: str) -> Tuple[Union[int, float], float]:
    vals = np.sort(data[attr].unique())
    best_gain = 0
    best_split_val = None
    for i in range(len(vals)-1):
        split_val = (vals[i] + vals[i + 1]) / 2
        if measure == 'ig':
            gain = information_gain(data, labels, attr, split_val)
        else:
            gain = gini_gain(data, labels, attr, split_val)
        if best_gain < gain:
            best_gain = gain
            best_split_val = split_val

    return (best_split_val, best_gain)


def split_data(df: pd.DataFrame, frac_1: float = 0.8, frac_2: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if abs((frac_1 + frac_2) - 1.0) > 1e-6:
        raise ValueError('Fractions should add up to 1')
    
    df = df.sample(frac=1.0).reset_index(drop=True)
    ind = int(len(df.index) * frac_1)
    df_1 = df.iloc[:ind, :].reset_index(drop=True)
    df_2 = df.iloc[ind:, :].reset_index(drop=True)
    return (df_1, df_2)


def split_df_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return (data, labels)


def get_pred_accuracy(tree: DecisionTree, test: pd.DataFrame) -> Tuple[pd.Series, float]:
    test_data, test_labels = split_df_col(test)
    preds = tree.predict(test_data)
    # accuracy = np.mean(preds.reset_index(drop=True) == test_labels.reset_index(drop=True)) * 100
    accuracy = np.mean(preds == test_labels) * 100
    return (preds, accuracy)


def save_plot(x: List, y: List, param: str) -> None:
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
    true_pos = np.sum(np.logical_and(labels, preds))
    return true_pos / (np.sum(preds) + 1e-6)


def recall(preds, labels):
    true_pos = np.sum(np.logical_and(labels, preds))
    return true_pos / (np.sum(labels) + 1e-6)


def f1_score(preds, labels):
    p = precision(preds, labels)
    r = recall(preds, labels)
    return 2 * (p * r) / (p + r + 1e-6)
