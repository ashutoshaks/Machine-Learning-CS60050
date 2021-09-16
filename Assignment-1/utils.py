import pandas as pd
from typing import Tuple, Union
import math

def entropy(labels: pd.Series) -> float:
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [-1*(x/total) * math.log2(x/total) for x in diff_vals]
    return sum(diff_vals)

def information_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    filt = data[attr] < split_val
    left_labels = pd.Series(labels[i] for i in df[~filt].index)
    right_labels = pd.Series(labels[i] for i in df[filt].index)
    gain = entropy(labels) - (len(left_labels) * entropy(left_labels) + len(right_labels) * entropy(right_labels))/len(labels)
    return gain

def gini_index(labels: pd.Series) -> float:
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [(x/total)**2 for x in diff_vals]
    return 1 - sum(diff_vals)

def gini_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    initial_gain = gini_index(labels)
    filt = data[attr] < split_val
    left_labels = pd.Series(labels[i] for i in df[~filt].index)
    right_labels = pd.Series(labels[i] for i in df[filt].index)
    final_gain = (len(left_labels) * gini_index(left_labels) + len(right_labels) * gini_index(right_labels)) / len(labels)
    return initial_gain - final_gain

def find_best_split(data: pd.DataFrame, labels: pd.Series, attr: str, measure: str) -> Tuple[Union[int, float], float]:
    sorted = data.sort_values(by=attr)[attr].tolist()
    best_gain = 0
    best_split_val = None
    if measure == INFORMATION_GAIN:
        for i in range(len(sorted)-1):
            split_val = (sorted[i]+sorted[i+1])/2
            gain = information_gain(data, labels, attr, split_val)
            if best_gain < gain:
                best_gain = gain
                best_split_val = split_val
    else:
        for i in range(len(sorted)-1):
            split_val = (sorted[i]+sorted[i+1])/2
            gain = gini_gain(data, labels, attr, split_val)
            if best_gain < gain:
                best_gain = gain
                best_split_val = split_val