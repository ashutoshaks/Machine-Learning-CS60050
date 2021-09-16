import pandas as pd
import graphviz
from typing import Dict, Tuple, Union
from utils import find_best_split

class Node:

    def __init__(self, attr: str, split_val: Union[int, float], prob_label: int) -> None:
        self.attr = attr
        self.split_val = split_val
        self.prob_label = prob_label
        self.left = None
        self.right = None

    def is_leaf(self) -> bool:
        return (self.left is None) and (self.right is None)

    def format_string(self) -> str:
        if self.is_leaf():
            outcome = "Yes" if self.prob_label == 1 else "No"
            return f"{self.attr}\n{outcome}"
        else:
            return f"{self.attr} < {self.split_val}"


class DecisionTree:

    def __init__(self, max_depth: int = 30, min_samples: int = 1, measure: str = 'gini') -> None:
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.measure = measure
        self.tree_depth = 0

    def build_tree(self, train_data: pd.DataFrame, train_labels: pd.Series, depth: int = 0) -> Node:
        if (depth == self.max_depth) or (len(train_data) <= self.min_samples) or (len(train_labels.unique()) == 1):
            return self.create_leaf(train_labels)

        attr, split_val = self.get_best_attribute(train_data, train_labels)
        node = Node(attr, split_val, train_labels.value_counts().idxmax())

        filt = train_data[attr] < split_val

        left_data = train_data[filt]
        left_labels = train_labels[filt]
        node.left = self.build_tree(left_data, left_labels, depth + 1)

        right_data = train_data[~filt]
        right_labels = train_labels[~filt]
        node.right = self.build_tree(right_data, right_labels, depth + 1)

        self.tree_depth = max(self.tree_depth, depth)

        return node

    def get_best_attribute(self, train_data: pd.DataFrame, train_labels: pd.Series) -> Tuple[str, Union[int, float]]:
        attributes = train_data.columns
        max_gain = -10**20
        best_attr = None
        best_split_val = None
        for attr in attributes:
            split_val, gain = find_best_split(train_data, train_labels, attr, self.measure)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
                best_split_val = split_val
        return (best_attr, best_split_val)


    def create_leaf(self, labels: pd.Series) -> Node:
        prob_label = labels.value_counts().idxmax()
        return Node('Outcome', None, prob_label)

    def predict_one(self, test_dict: Dict, root: Node) -> int:
        if root is None:
            return None
        if root.is_leaf():
            return root.prob_label

        if test_dict[root.attr] < root.split_val:
            return self.predict_one(test_dict, root.left)
        else:
            return self.predict_one(test_dict, root.right)

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        predictions = pd.Series([self.predict_one(row, self.root) for row in test_data.to_dict(orient='records')])
        return predictions

    def get_pred_accuracy(self, test_data: pd.DataFrame, test_labels: pd.Series) -> Tuple[pd.Series, float]:
        preds = self.predict(test_data)
        comp = (preds == test_labels)
        accuracy = comp.astype('int32').mean() * 100.0
        return (preds, accuracy)

    def print_tree(self, file: str) -> None:
        tree = graphviz.Digraph(
            filename=file,
            graph_attr={
                'rankdir': 'LR'
            },
            node_attr={
                'shape': 'Rectangle'
            }
        )

        root = self.root
        queue = []
        queue.append(root)
        root.id = 0
        tree.node(str(root.id), label=root.format_string())
        uid = 1
        edge_labels = ['True', 'False']
        
        while(len(queue) > 0):
            node = queue.pop(0)
            for i, child in enumerate([node.left, node.right]):
                if child != None:
                    child.id = uid
                    uid += 1
                    queue.append(child)
                    tree.node(str(child.id), label=child.format_string())
                    tree.edge(str(node.id), str(child.id), label=edge_labels[i])

        tree.render(file, view=True)
        