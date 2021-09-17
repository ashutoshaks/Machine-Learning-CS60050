import argparse
from typing import List, Tuple
import pandas as pd
from decision_tree import DecisionTree
from utils import f1_score, split_data, save_plot, split_df_col, get_pred_accuracy


def dt_utility(train: pd.DataFrame, test: pd.DataFrame, max_depth: int, measure: str):
    tree = DecisionTree(measure, max_depth)
    tree.train(train)
    _, train_acc = get_pred_accuracy(tree, train)
    _, test_acc = get_pred_accuracy(tree, test)

    return (tree, train_acc, test_acc)


def compare_measures(train: pd.DataFrame, test: pd.DataFrame, max_depth: int) -> None:
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


def select_best_tree(train: pd.DataFrame, test: pd.DataFrame, max_depth: int, measure: str, num_splits: int = 10):
    avg_train_acc, avg_test_acc, best_test_acc = 0, 0, 0
    best_tree, best_train, best_valid = None, None, None

    for i in range(num_splits):
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


def vary_depth_nodes(df: pd.DataFrame, measure: str) -> None:
    node_dict = dict()
    d_lim = 15
    depths = []
    trees = [None] * (d_lim + 1)
    max_acc = [0] * (d_lim + 1)
    acc = [0] * (d_lim + 1)
    for iter in range(10):
        print(f'Iteration {iter + 1}')
        train, test = split_data(df, 0.8, 0.2)
        for d in range(1, d_lim + 1):
            tree, train_acc, test_acc = dt_utility(train, test, d, measure)
            if test_acc > max_acc[d]:
                trees[d] = tree
            count = tree.root.node_count()
            node_dict[count] = test_acc
            print(f'Depth: {d}, Train Accuracy: {train_acc:.4f}% Average Accuracy: {test_acc:.4f}%')
            acc[d] += test_acc
        print()

    for i in range(1, d_lim + 1):
        depths.append(i)
    acc = [x / 10 for x in acc]
    print(acc)
    best_depth = acc.index(max(acc))
    trees[best_depth].print_tree('best_depth.gv')
    print(f'Best Depth: {best_depth}')
    print(f'Accuracy for depth {best_depth}: {max(acc):.4f}%')
    save_plot(depths, acc[1:], 'depth')
    lists = sorted(node_dict.items())
    x, y = zip(*lists)
    save_plot(x, y, 'nodes')


def main():
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
        raise ValueError('Measure must be either')
    file = args.file

    df = pd.read_csv(file)
    print('\n------------ LOADED DATA ------------\n')

    train, test = split_data(df, 0.8, 0.2)

    print('\n------------ PART 1 - COMPARING IMPURITY MEASURES ------------\n')
    compare_measures(train, test, 5)

    print('\n------------ PART 2 - DETERMINING ACCURACY OVER 10 RANDOM SPLITS ------------\n')
    tree, avg_test_acc, _, best_test_acc, train, valid = select_best_tree(train, test, max_depth, measure, 10)
    print(f'Average Test Accuracy: {avg_test_acc:.4f}%')
    print(f'Best Test Accuracy: {best_test_acc:.4f}%')
    tree.print_tree('before_pruning.gv')

    print('\n------------ PART 3 - DETERMINING ACCURACY BY VARYING MAXIMUM DEPTH ------------\n')
    vary_depth_nodes(df, measure)

    print('\n------------ PART 4 - PRUNING THE TREE OBTAINED FROM PART 2 ------------\n')
    _, valid_acc = get_pred_accuracy(tree, valid)
    tree.root.prune(tree, valid_acc, valid)

    _, test_acc = get_pred_accuracy(tree, test)
    print(f'Test accuracy after pruning: {test_acc:.4f}%')
    tree.print_tree('after_pruning.gv')


if __name__ == '__main__':
    main()
