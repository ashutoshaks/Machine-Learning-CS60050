import argparse
from typing import Tuple
import pandas as pd
from decision_tree import DecisionTree
from utils import split_data, save_plot, split_df_col


def compare_measures(df: pd.DataFrame, max_depth: int) -> None:
    train, test = split_data(df, 0.8, 0.2)
    train_data, train_labels = split_df_col(train)
    test_data, test_labels = split_df_col(test)
    measures = ['ig', 'gini']

    for measure in measures:
        tree = DecisionTree(measure, max_depth)
        tree.train(train_data, train_labels)
        _, train_acc = tree.get_pred_accuracy(train_data, train_labels)
        _, test_acc = tree.get_pred_accuracy(test_data, test_labels)
        measure_name = 'Information Gain' if measure == 'ig' else 'Gini Index'
        print(
            f'Impurity Measure: {measure_name}\nTrain Accuracy: {train_acc}%, Test Accuracy: {test_acc}%')


def select_best_tree(df: pd.DataFrame, max_depth: int, measure: str, num_splits: int = 10) -> Tuple[float, float, DecisionTree, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    avg_test_acc, best_test_acc = 0, 0
    best_tree, best_train, best_valid, best_test = None, None, None, None

    for i in range(num_splits):
        train, test = split_data(df, 0.8, 0.2)
        train, valid = split_data(train, 0.75, 0.25)
        train_data, train_labels = split_df_col(train)
        test_data, test_labels = split_df_col(test)

        tree = DecisionTree(measure, max_depth)
        tree.train(train_data, train_labels)

        _, test_acc = tree.get_pred_accuracy(test_data, test_labels)
        print(f'Split {i + 1} - Test Accuracy: {test_acc}%')
        avg_test_acc += test_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_tree, best_train, best_valid, best_test = tree, train, valid, test

    avg_test_acc /= num_splits
    return (avg_test_acc, best_test_acc, best_tree, best_train, best_valid, best_test)


def vary_depth(df: pd.DataFrame, measure: str) -> None:
    best_avg_acc = 0
    best_depth = None
    depths = []
    acc = []
    for d in range(1, 51):
        tree, avg_acc, max_acc = select_best_tree(df, d, measure)
        print(f'Depth: {d}, Average Accuracy: {avg_acc}%')
        depths.append(d)
        acc.append(avg_acc)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_depth = d

    print(f'Best Depth: {best_depth}')
    print(f'Accuracy for depth {best_depth}: {best_avg_acc}%')
    save_plot(depths, acc, 'depth')


def vary_num_nodes(df: pd.DataFrame, measure: str) -> None:
    pass


def main():
    parser = argparse.ArgumentParser(description='Decision Tree Algorithm')
    parser.add_argument('--depth', type=int,
                        default=15,
                        help='maximum depth for Q1 and Q2 (default: 15)')
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
        raise ValueError('Measure must be either')
    file = args.file

    df = pd.read_csv(file)
    print('\n------------ LOADED DATA ------------\n')

    print('\n------------ PART 1 - COMPARING IMPURITY MEASURES ------------\n')
    compare_measures(df, max_depth)

    print('\n------------ PART 2 - DETERMINING ACCURACY OVER 10 RANDOM SPLITS ------------\n')
    tree, avg_acc, best_acc, train, valid, test = select_best_tree(df, max_depth, measure)
    print(f'Average Test Accuracy: {avg_acc}%')
    print(f'Best Test Accuracy: {best_acc}%')

    print('\n------------ PART 3 - DETERMINING ACCURACY BY VARYING MAXIMUM DEPTH ------------\n')
    vary_depth(df, measure)


if __name__ == '__main__':
    main()
