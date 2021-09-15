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