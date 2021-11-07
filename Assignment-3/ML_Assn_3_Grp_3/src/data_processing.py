# Machine Learning - Assignment 3
# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008

import pandas as pd
import numpy as np

def read_data():
    """
    Reads the data from the csv file and returns the data as numpy arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The feature vector and the labels.
    """
    df_1 = pd.read_csv('../occupancy_data/datatraining.txt')
    df_2 = pd.read_csv('../occupancy_data/datatest.txt')
    df_3 = pd.read_csv('../occupancy_data/datatest2.txt')

    df = pd.concat([df_1, df_2, df_3])
    df.drop('date', axis=1, inplace=True)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y


def split_data(X, y, train_ratio = 0.7, valid_ratio = 0.1, test_ratio = 0.2):
    """
    Splits the data into training, validation and test sets.

    Args:
        X (np.ndarray): The feature vector.

        y (np.ndarray): The labels.

        train_ratio (float, optional):  The fraction of examples in the training set. Defaults to 0.7.

        valid_ratio (float, optional): The fraction of examples in the validation set. Defaults to 0.1.

        test_ratio (float, optional): The fraction of examples in the test set. Defaults to 0.2.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            The training set, the training labels, the validation set, the validation labels, the test set and the test labels.
    """
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    ind_1 = int(train_ratio * len(X) + 0.5)
    ind_2 = int(ind_1 + valid_ratio * len(X))
    X_train, y_train = X[:ind_1], y[:ind_1]
    X_valid, y_valid = X[ind_1:ind_2], y[ind_1:ind_2]
    X_test, y_test = X[ind_2:], y[ind_2:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def normalize(X_train, X_valid, X_test):
    """
    Normalizes the data to have zero mean and unit variance.
    Note that we normalize the validation and test sets using the mean and standard deviation of the training set.

    Args:
        X_train (np.ndarray): The training set.

        X_valid (np.ndarray): The validation set.

        X_test (np.ndarray): The test set.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            The normalized training set, the normalized validation set and the normalized test set.
    """
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_valid = (X_valid - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_valid, X_test
