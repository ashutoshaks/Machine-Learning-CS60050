import pandas as pd
import graphviz
from utils import find_best_split, split_df_col, get_pred_accuracy


class DecisionTree:     # Forward declaration
    pass


class Node:
    """
    The Node class for our decision tree.
    It contains the attributes required for each node and functions for various tasks.
    """

    def __init__(self, attr, split_val, prob_label):
        """
        Initializes a node with proper values.

        Args:
            attr (str): The decision attribute selected for the node
                    on the basis of which we split the tree further.

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


    def is_leaf(self):
        """
        Checks if the given node is a leaf.

        Returns:
            bool: True, if the node is a leaf, otherwise False.
        """
        return (self.left is None) and (self.right is None)

    
    def node_count(self):
        """
        Finds the number of nodes in the subtree rooted at the given node.

        Returns:
            int: Number of nodes in the subtree.
        """
        left_count, right_count = 0, 0
        if self.left != None:
            left_count = self.left.node_count()         # recurse on left subtree
        if self.right != None:
            right_count = self.right.node_count()       # recurse on right subtree
        return 1 + left_count + right_count
        

    def prune(self, tree, accuracy, valid):
        """
        Prunes a node by recursively first pruning the subtree of this node,
        and then checks if the current node can be pruned. Pruning continues till
        the accuracy on the validation set increases.

        Args:
            tree (DecisionTree): The decision tree that is being pruned.

            accuracy (float): The best accuracy on the validation set that can 
                    be achieved till now.

            valid (pd.DataFrame): The validation set used for calculating accuracy
                    while pruning.

        Returns:
            float: If pruning the current node increases the accuracy, then return the 
                    updated accuracy after pruning this node, else return the original
                    accuracy.
        """
        new_acc = 0

        # cannot prune leaf nodes
        if self.left is None and self.right is None:
            return accuracy

        if self.left != None:
            new_acc = self.left.prune(tree, accuracy, valid)        # prune left subtree
        if self.right != None:
            new_acc = self.right.prune(tree, new_acc, valid)        # prune right subtree

        # temporarily store left and right children
        left, right = self.left, self.right

        # temprarily remove the children of the current node
        self.left = None
        self.right = None

        _, temp_acc = get_pred_accuracy(tree, valid)

        # decide if we will prune this node or node
        if temp_acc < new_acc or tree.root.node_count() <= 5:
            self.left, self.right = left, right
        else:
            new_acc = temp_acc
            self.attr = 'Outcome'

        return new_acc


    def format_string(self):
        """
        Generates the string to be displayed in each node while printing the tree.

        Returns:
            str: A string to be displayed, depending on whether the node is a leaf or not.
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

    def __init__(self, measure='ig', max_depth=10, min_samples=1):
        """
        Initializes a decision tree with proper metadata.

        Args:
            measure (str, optional): The impurity measure to be used - information gain,
                    or gini index. Defaults to 'ig'.

            max_depth (int, optional): Maxmimum depth of the decision tree. Defaults to 15.

            min_samples (int, optional): Minimum number of samples that must be present to
                    branch the tree further. Defaults to 1.
        """
        self.root = None
        self.measure = measure
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree_depth = 0


    def train(self, train):
        """
        Trains the decision tree model.

        Args:
            train (pd.DataFrame): The training dataset.
        """
        train_data, train_labels = split_df_col(train)
        self.root = self.build_tree(train_data, train_labels)
        

    def build_tree(self, train_data, train_labels, depth=0):
        """
        Builds the entire decision tree recursively, by deciding which attribute
        to split on, divides the data into two parts, and then calls the same
        function for the left and right subtrees.

        Args:
            train_data (pd.DataFrame): The training dataset without the output labels.

            train_labels (pd.Series): The output labels for each row in the training dataset.

            depth (int, optional): Depth of the current node. Defaults to 0.

        Returns:
            Node: Root node of the tree
        """

        # if maximum depth is reached, or if we do not have enough samples, or all the samples have the same outcome label
        # then make this node a leaf
        if (depth == self.max_depth) or (len(train_data) <= self.min_samples) or (len(train_labels.unique()) == 1):
            return self.create_leaf(train_labels)

        attr, split_val = self.get_best_attribute(train_data, train_labels)
        node = Node(attr, split_val, train_labels.value_counts().idxmax())

        # create left partition for data[attr] < split_val
        filt = train_data[attr] < split_val
        left_data = train_data[filt]
        left_labels = train_labels[filt]
        node.left = self.build_tree(left_data, left_labels, depth + 1)

        # create right partition for data[attr] >= split_val
        right_data = train_data[~filt]
        right_labels = train_labels[~filt]
        node.right = self.build_tree(right_data, right_labels, depth + 1)

        self.tree_depth = max(self.tree_depth, depth)
        return node


    def get_best_attribute(self, train_data, train_labels):
        """
        Finds the best attribute on the basis of which splitting should be done.

        Args:
            train_data (pd.DataFrame): The training dataset without the output labels.

            train_labels (pd.Series): The output labels for each row in the training dataset.

        Returns:
            Tuple[str, Union[int, float]]: The best attribute found, and the corresponding
                    value of that attribute around which we should split.
        """
        attributes = train_data.columns
        max_gain = -10**20
        best_attr = None
        best_split_val = None

        # iterate on all attributes to choose the best one
        for attr in attributes:
            split_val, gain = find_best_split(train_data, train_labels, attr, self.measure)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
                best_split_val = split_val
        return (best_attr, best_split_val)


    def create_leaf(self, labels):
        """
        Creates and returns a leaf node for the decision tree.

        Args:
            labels (pd.Series): The output labels of the data points at this node.

        Returns:
            Node: The leaf node created.
        """
        prob_label = labels.value_counts().idxmax()
        return Node('Outcome', None, prob_label)


    def predict_one(self, test_dict, root):
        """
        Predicts the outcome of a single sample on the basis of the decision tree created.

        Args:
            test_dict (Dict): A dictionary containing the attributes and corresponding values
                    of the data point.

            root (Node): Current node while traversing the tree for finding the predicted outcome.

        Returns:
            int: 0 or 1, the prediction found from the decision tree.
        """
        if root is None:
            return None
        if root.is_leaf():
            return root.prob_label
        if test_dict[root.attr] < root.split_val:
            return self.predict_one(test_dict, root.left)
        else:
            return self.predict_one(test_dict, root.right)


    def predict(self, test_data):
        """
        Predicts the outcome on a set of test data.

        Args:
            test_data (pd.DataFrame): The test dataset for which predictions are to be made.

        Returns:
            pd.Series: Predicted outcomes (series of 0, 1 values) for the test dataset.
        """
        predictions = pd.Series([self.predict_one(row, self.root) for row in test_data.to_dict(orient='records')])
        return predictions


    def print_tree(self, file):
        """
        Prints the decision tree in a neat format using the graphviz library.

        Args:
            file (str): File name with which the image of the tree is to be saved. 
        """
        tree = graphviz.Digraph(filename=file, format='png', node_attr={'shape': 'box'})
        root = self.root
        queue = []
        queue.append(root)
        root.id = 0
        tree.node(str(root.id), label=root.format_string())
        uid = 1
        edge_labels = ['True', 'False']
        
        # print the tree using a breadth first search
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
