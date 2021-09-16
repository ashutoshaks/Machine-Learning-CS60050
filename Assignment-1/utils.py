import pandas as pd
from typing import Tuple, Union

def entropy(labels: pd.Series) -> float:
    pass

def information_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    pass

def gini_index(labels: pd.Series) -> float:
    pass

def gini_gain(data: pd.DataFrame, labels: pd.Series, attr: str, split_val: Union[int, float]) -> float:
    pass

def find_best_split(data: pd.DataFrame, labels: pd.Series, attr: str, measure: str) -> Tuple[Union[int, float], float]:
    pass

def split_data(df: pd.DataFrame, frac_1: float = 0.8, frac_2: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if abs((frac_1 + frac_2) - 1.0) > 1e-6:
        raise ValueError('Fractions should add up to 1')
    
    df = df.sample(1.0)
    df_1 = df.sample(frac=frac_1)
    df_2 = df.drop(df_1.index).sample(frac=1.0)

    return (df_1, df_2)

def split_df_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    return (data, labels)
