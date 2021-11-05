from typing import Tuple
import pandas as pd
import numpy as np

def read_data() -> None:
    df_1 = pd.read_csv('occupancy_data/datatraining.txt')
    df_2 = pd.read_csv('occupancy_data/datatest.txt')
    df_3 = pd.read_csv('occupancy_data/datatest2.txt')

    df = pd.concat([df_1, df_2, df_3])
    df.drop('date', axis=1, inplace=True)
    # df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    # df['date'] = df['date'].apply(lambda date: (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, valid_ratio: float = 0.1,
                test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    ind_1 = int(train_ratio * len(X) + 0.5)
    ind_2 = int(ind_1 + valid_ratio * len(X))
    X_train, y_train = X[:ind_1], y[:ind_1]
    X_valid, y_valid = X[ind_1:ind_2], y[ind_1:ind_2]
    X_test, y_test = X[ind_2:], y[ind_2:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def normalize(X_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_valid = (X_valid - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_valid, X_test
