import argparse
from typing import List, Tuple
import pandas as pd
from decision_tree import DecisionTree
from utils import split_data, save_plot, split_df_col

import datetime

def dt_utility(train: pd.DataFrame, test: pd.DataFrame, max_depth: int, measure: str):
    tree = DecisionTree(measure, max_depth)
    tree.train(train)
    _, train_acc = tree.get_pred_accuracy(train)
    _, test_acc = tree.get_pred_accuracy(test)

    return (tree, train_acc, test_acc)


def compare_measures(train: pd.DataFrame, test: pd.DataFrame, max_depth: int) -> None:
    measures = ['ig', 'gini']
    # measures = ['gini']

    for measure in measures:
        tree, train_acc, test_acc = dt_utility(train, test, max_depth, measure)
        measure_name = 'Information Gain' if measure == 'ig' else 'Gini Index'
        print(f'Impurity Measure: {measure_name}\nTrain Accuracy: {train_acc}%, Test Accuracy: {test_acc}%')
        tree.print_tree(f'{measure}.gv')


def select_best_tree(train: pd.DataFrame, test: pd.DataFrame, max_depth: int, measure: str, num_splits: int = 10):
    avg_train_acc, avg_test_acc, best_test_acc = 0, 0, 0
    best_train_acc = 0
    best_tree, best_train, best_valid = None, None, None

    for i in range(num_splits):
        train, valid = split_data(train, 0.75, 0.25)
        tree, train_acc, test_acc = dt_utility(train, test, max_depth, measure)
        
        print(f'Split {i + 1} - Test Accuracy: {test_acc}%')
        avg_test_acc += test_acc
        avg_train_acc += train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_tree, best_train, best_valid = tree, train, valid

    avg_test_acc /= num_splits
    avg_train_acc /= num_splits
    return (best_tree, avg_test_acc, avg_train_acc, best_test_acc, best_train, best_valid)


def vary_depth(train: pd.DataFrame, test: pd.DataFrame, measure: str) -> None:
    best_avg_acc = 0
    best_depth = None
    depths = []
    acc = []
    for d in range(1, 16):
        tree, avg_test_acc, avg_train_acc, _, _, _, _ = select_best_tree(train, test, measure, d, num_splits=2)
        print(f'Depth: {d}, Train Accuracy: {avg_train_acc}% Average Accuracy: {avg_test_acc}%')
        depths.append(d)
        acc.append(avg_test_acc)
        if avg_test_acc > best_avg_acc:
            best_avg_acc = avg_test_acc
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
                        default='gini',
                        help='impurity measure for Q2 and Q3 ("ig" = information gain, "gini" = gini index) (default: gini)')
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

    train, test = split_data(df, 0.8, 0.2)

    for i in range(9, 10):
        start = datetime.datetime.now()
        print(f'depth = {i}')
        tree, test_acc, train_acc, _, best_train, best_valid = select_best_tree(train, test, i, 'gini', 1)
        print(f'Train Accuracy: {train_acc}%, Test Accuracy: {test_acc}%')
        tree.print_tree('before.gv')

        _, valid_acc = tree.get_pred_accuracy(best_valid)
        print(f'Validation Accuracy: {valid_acc}%')
        tree.root.prune(tree, valid_acc, best_valid)
        tree.print_tree('after.gv')
        print('After pruning...')
        _, valid_acc = tree.get_pred_accuracy(best_valid)
        print(f'Validation Accuracy: {valid_acc}%')
        _, t_acc = tree.get_pred_accuracy(test)
        print(f'Test Accuracy: {t_acc}%')
        end = datetime.datetime.now()
        diff = end - start
        print(diff.total_seconds() * 1000)
        print()


    # # print('\n------------ PART 1 - COMPARING IMPURITY MEASURES ------------\n')
    # compare_measures(train, test, max_depth)

    # print('\n------------ PART 2 - DETERMINING ACCURACY OVER 10 RANDOM SPLITS ------------\n')
    # tree, avg_test_acc, _, best_test_acc, train, valid = select_best_tree(train, test, max_depth, measure)
    # print(f'Average Test Accuracy: {avg_test_acc}%')
    # print(f'Best Test Accuracy: {best_test_acc}%')

    # # print('\n------------ PART 3 - DETERMINING ACCURACY BY VARYING MAXIMUM DEPTH ------------\n')
    # vary_depth(train, test, measure)


if __name__ == '__main__':
    main()
