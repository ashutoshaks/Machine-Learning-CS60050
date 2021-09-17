import argparse
import pandas as pd
from decision_tree import DecisionTree
from utils import f1_score, split_data, save_plot, split_df_col, get_pred_accuracy, dt_utility


def compare_measures(train, test, max_depth):
    """
    Compares the impact of the two impurity measures - information gain and gini index.

    Args:
        train (pd.DataFrame): The training dataset.

        test (pd.DataFrame): The test dataset.

        max_depth (int): The maximum depth allowed for the decision tree created.
    """
    measures = ['ig', 'gini']

    for measure in measures:
        tree = DecisionTree(measure, max_depth)
        tree.train(train)
        _, train_acc = get_pred_accuracy(tree, train)
        preds, test_acc = get_pred_accuracy(tree, test)
        _, labels = split_df_col(test)
        f1 = f1_score(preds, labels)
        
        measure_name = 'Information Gain' if measure == 'ig' else 'Gini Index'
        print(f'Impurity Measure: {measure_name}\nTrain Accuracy: {train_acc:.4f}%, Test Accuracy: {test_acc:.4f}%, f1 score: {f1:.4f}\n')


def select_best_tree(train, test, max_depth, measure='ig', num_splits=10):
    """
    Generated 10 decision trees based on 10 random 80/20 splits of the data, finds the 
    average accuracy and returns the one with the best test accuracy.

    Args:
        train (pd.DataFrame): The training dataset.

        test (pd.DataFrame): The test dataset.

        max_depth (int): The maximum depth allowed for the decision trees.

        measure (str): The impurity measure to be used. Defaults to 'ig'.

        num_splits (int, optional): The number of random splits. Defaults to 10.

    Returns:
        Tuple[DecisionTree, float, float, float, pd.DataFrame, pd.DataFrame]: The decision tree
                with best test accuracy, the average test accuracy, the average train accuracy,
                the best test accuracy, the training and validation set partition for the 
                best split.
    """
    avg_train_acc, avg_test_acc, best_test_acc = 0, 0, 0
    best_tree, best_train, best_valid = None, None, None

    for i in range(num_splits):
        # split into training and validation set, validation set will be used later during pruning
        train_df, valid = split_data(train, 0.75, 0.25)
        tree, train_acc, test_acc = dt_utility(train_df, test, max_depth, measure)
        
        print(f'Split {i + 1} - Test Accuracy: {test_acc:.4f}%')
        avg_test_acc += test_acc
        avg_train_acc += train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_tree, best_train, best_valid = tree, train_df, valid

    avg_test_acc /= num_splits
    avg_train_acc /= num_splits
    return (best_tree, avg_test_acc, avg_train_acc, best_test_acc, best_train, best_valid)


def vary_depth_nodes(df, measure='ig'):
    """
    Analyzes the effect of variation in depth, and number of nodes on the performance 
    of the decision tree. If we try to vary just the number of nodes independently, 
    then that will be computationally very expensive. So, we vary the depth, and observe 
    10 splits for the depth and get 10 decision trees for each depth, and for each such 
    decision tree, we record the number of nodes and the corresponding accuracy.

    Args:
        df (pd.DataFrame): The entire dataframe (dataset) available to us.

        measure (str): The impurity measure to be used. Defaults to 'ig'.
    """
    node_dict = dict()      # dictionary to store (no. of nodes, accuracy) pairs
    d_lim = 15              # maximum depth upto which we shall vary
    depths = []
    trees = [None] * (d_lim + 1)    # list for storing the best trees for each depth
    max_acc = [0] * (d_lim + 1)     # list for storing the best accuracy for each depth
    acc = [0] * (d_lim + 1)

    # iterate 10 times for each depth to get better results
    for iter in range(10):
        train, test = split_data(df, 0.8, 0.2)
        for d in range(1, d_lim + 1):
            tree, train_acc, test_acc = dt_utility(train, test, d, measure)
            if test_acc > max_acc[d]:
                trees[d] = tree
                max_acc[d] = test_acc

            # add (node count, accuracy) pair to the dictionary
            count = tree.root.node_count()
            node_dict[count] = test_acc

            acc[d] += test_acc

    acc = [x / 10 for x in acc]

    # populate the depths list and print accuracy for each depth
    for i in range(1, d_lim + 1):
        depths.append(i)
        print(f'Depth: {i}, Average Accuracy: {acc[i]:.4f}%')
    best_depth = acc.index(max(acc))

    # print the tree for best depth 
    trees[best_depth].print_tree('best_depth.gv')

    print(f'Best Depth: {best_depth}')
    print(f'Accuracy for depth {best_depth}: {max(acc):.4f}%')

    # save depths v/s accuracy and node count v/s accuracy plots
    save_plot(depths, acc[1:], 'depth')
    lists = sorted(node_dict.items())
    x, y = zip(*lists)
    save_plot(x, y, 'nodes')


def main():
    """
    The main function that performs all the tasks required according to the problem statement.
    """
    parser = argparse.ArgumentParser(description='Decision Tree Algorithm')
    parser.add_argument('--depth', type=int,
                        default=10,
                        help='maximum depth for Q1 and Q2 (default: 10)')
    parser.add_argument('--measure', type=str,
                        default='ig',
                        help='impurity measure for Q2 and Q3 ("ig" = information gain, "gini" = gini index) (default: ig)')
    parser.add_argument('--file', type=str,
                        default='diabetes.csv',
                        help='file name with path for data file (default: diabetes.csv)')
    args = parser.parse_args()

    max_depth = args.depth
    if max_depth <= 0:
        raise ValueError("Maximum depth must be positive")
    measure = args.measure
    if measure != 'ig' and measure != 'gini':
        raise ValueError('Measure must be either "ig" or "gini"')
    file = args.file

    # read the data from the csv file into a dataframe
    df = pd.read_csv(file)
    print('\n------------ LOADED DATA ------------\n')

    # split the data into train and test set
    train, test = split_data(df, 0.8, 0.2)

    # solve Q1
    print('\n------------ PART 1 - COMPARING IMPURITY MEASURES ------------\n')
    compare_measures(train, test, 5)

    # solve Q2
    print('\n------------ PART 2 - DETERMINING ACCURACY OVER 10 RANDOM SPLITS ------------\n')
    tree, avg_test_acc, _, best_test_acc, train, valid = select_best_tree(train, test, max_depth, measure, 10)
    print(f'Average Test Accuracy: {avg_test_acc:.4f}%')
    print(f'Best Test Accuracy: {best_test_acc:.4f}%')
    tree.print_tree('before_pruning.gv')

    # solve Q3
    print('\n------------ PART 3 - DETERMINING ACCURACY BY VARYING MAXIMUM DEPTH ------------\n')
    vary_depth_nodes(df, measure)

    # solve Q4
    print('\n------------ PART 4 - PRUNING THE TREE OBTAINED FROM PART 2 ------------\n')
    _, valid_acc = get_pred_accuracy(tree, valid)
    tree.root.prune(tree, valid_acc, valid)

    _, test_acc = get_pred_accuracy(tree, test)
    print(f'Test accuracy after pruning: {test_acc:.4f}%')
    tree.print_tree('after_pruning.gv')


if __name__ == '__main__':
    main()
