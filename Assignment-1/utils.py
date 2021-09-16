import pandas as pd
import math
from typing import List, Tuple, Union
from matplotlib import pyplot as plt

RANDOM_SEED = 100

def entropy(labels: pd.Series) -> float:
    total = len(labels)
    diff_vals = labels.value_counts().tolist()
    diff_vals = [-1*(x/total) * math.log2(x/total) for x in diff_vals]
    return sum(diff_vals)

def information_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    filt = data[attr] < split_val
    left_labels = labels[filt]
    right_labels = labels[filt]
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
    left_labels = labels[filt]
    right_labels = labels[filt]
    final_gain = (len(left_labels) * gini_index(left_labels) + len(right_labels) * gini_index(right_labels)) / len(labels)
    return initial_gain - final_gain

def find_best_split(data: pd.DataFrame, labels: pd.Series, attr: str, measure: str) -> Tuple[Union[int, float], float]:
    sorted = data.sort_values(by=attr)[attr].tolist()
    best_gain = 0
    best_split_val = None
    for i in range(len(sorted)-1):
        split_val = (sorted[i]+sorted[i+1])/2
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
    
    df = df.sample(1.0, random_state=RANDOM_SEED)
    df_1 = df.sample(frac=frac_1, random_state=RANDOM_SEED)
    df_2 = df.drop(df_1.index).sample(frac=1.0, random_state=RANDOM_SEED)

    return (df_1, df_2)

def split_df_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    return (data, labels)

# def split_data(df: pd.DataFrame, frac_1: float = 0.8, frac_2: float = 0.2) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
#     if abs((frac_1 + frac_2) - 1.0) > 1e-6:
#         raise ValueError('Fractions should add up to 1')
#     train, test = split_train_test(df, frac_1, frac_2)
#     train_data, train_labels = split_df_col(train)
#     test_data, test_labels = split_df_col(test)
#     return (train_data, train_labels, test_data, test_labels)

def save_plot(x: List, y: List, param: str) -> None:
    label = ('Depth' if param == 'depth' else 'No. of Nodes')
    plt.title(f'Test Accuracy v/s {label}')
    plt.xlabel(label)
    plt.ylabel('Test Accuracy')
    plt.plot(x, y)
    plt.savefig(f'{param}_acc.png')
