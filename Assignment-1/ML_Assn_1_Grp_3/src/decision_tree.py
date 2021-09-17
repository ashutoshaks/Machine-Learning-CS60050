import pandas as pd
import graphviz
from typing import Dict, Tuple, Union
from utils import find_best_split, split_df_col, get_pred_accuracy


class DecisionTree:
    pass


class Node:
    """
    The Node class for our decision tree
    It contains the attributes required for each node and functions for various tasks
    """

    def __init__(self, attr: str, split_val: Union[int, float], prob_label: int) -> None:
        """
        Initializes a node with proper values

        Args:
            attr (str): The decision attribute selected for the node
                    on the basis of which we split the tree further

            split_val (Union[int, float]): The value on whose basis splitting is done.
                    All data points with value of attr < split_val go to the left subtree,
                    and all with value >= split_val go to the right subtree.

            prob_label (int): This is the most probable outcome if we were to convert this 
                    node to a leaf. It is calculated by determining which outcome (0 or 1) 
                    occurs the most in the data points we have at this node.
        """
        self.attr = attr
        self.split_val = split_val
        self.prob_label = prob_label
        self.left = None
        self.right = None


    def is_leaf(self) -> bool:
        """
        Checks if the given node is a leaf

        Returns:
            bool: True, if the node is a leaf, otherwise False
        """
        return (self.left is None) and (self.right is None)

    
    def node_count(self) -> int:
        """
        Finds the number of nodes in the subtree rooted at the given node

        Returns:
            int: Number of nodes in the subtree
        """
        left_count, right_count = 0, 0
        if self.left != None:
            left_count = self.left.node_count()
        if self.right != None:
            right_count = self.right.node_count()
        return 1 + left_count + right_count
        

    def prune(self, tree: DecisionTree, accuracy: float, valid: pd.DataFrame) -> float:
        """
        Prunes a node by recursively first pruning the subtree of this node,
        and then checks if the cuurent node can be pruned. Pruning continues till
        the accuracy on the validation set increases

        Args:
            tree (DecisionTree): The decision tree that is being pruned

            accuracy (float): The best accuracy on the validation set that can 
                    be achieved till now

            valid (pd.DataFrame): The validation set used for calculating accuracy
                    while pruning

        Returns:
            float: If pruning the current node increases the accuracy, then return the 
                    updated accuracy after pruning this node, else return the original
                    accuracy
        """
        new_acc = 0
        if self.left is None and self.right is None:
            return accuracy
        if self.left != None:
            new_acc = self.left.prune(tree, accuracy, valid)
        if self.right != None:
            new_acc = self.right.prune(tree, new_acc, valid)

        left, right = self.left, self.right
        self.left = None
        self.right = None

        _, temp_acc = get_pred_accuracy(tree, valid)

        if temp_acc < new_acc or tree.root.node_count() <= 5:
            self.left, self.right = left, right
        else:
            new_acc = temp_acc
            self.attr = 'Outcome'

        return new_acc


    def format_string(self) -> str:
        """
        Generates the string to be displayed in each node while printing the tree

        Returns:
            str: A string to be displayed, depending on whether the node is a leaf or not
        """
        if self.is_leaf():
            outcome = 'Yes' if self.prob_label == 1 else 'No'
            return f'{self.attr}\n{outcome}'
        else:
            if self.attr == 'DiabetesPedigreeFunction':
                return f'{self.attr} <\n {self.split_val:.4f}'
            elif self.attr == 'BMI':
                return f'{self.attr} < {self.split_val:.4f}'
            else:
                return f'{self.attr} < {self.split_val}'


class DecisionTree:
    """
    The main Decision Tree class having metadata for the decision tree, and functions
    for various operations of the decision tree.
    """

    def __init__(self, measure: str = 'gini', max_depth: int = 15, min_samples: int = 1) -> None:
        """
        Initializes a decision tree with proper metadata

        Args:
            measure (str, optional): The impurity measure to be used - information gain,
                    or gini index. Defaults to 'gini'.

            max_depth (int, optional): Maxmimum depth of the decision tree. Defaults to 15.

            min_samples (int, optional): Minimum number of samples that must be present to
                    branch the tree further. Defaults to 1.
        """
        self.root = None
        self.measure = measure
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree_depth = 0


    def train(self, train: pd.DataFrame) -> None:
        train_data, train_labels = split_df_col(train)
        self.root = self.build_tree(train_data, train_labels)
        

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


    def print_tree(self, file: str) -> None:
        tree = graphviz.Digraph(filename=file, format='png', node_attr={'shape': 'box'})
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
