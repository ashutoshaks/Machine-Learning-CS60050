# Machine Learning - Assignment 2
# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008

import time
from data_processing import process_data, train_test_split
from naive_bayes import NaiveBayes
from metrics import display_metrics

def main():
    start = time.time()

    # Solve Part 1
    print('\n----------- PART 1 - LOADING AND PROCESSING DATA -----------\n')

    M, labels, label_map = process_data('train.csv')

    print(f'Shape of the binary feature matrix M: {M.shape}')
    print(f'Total number of examples: {M.shape[0]}')
    print(f'Vocabulary size: {M.shape[1]}')

    end_1 = time.time()
    print(f'\nTime taken for Part 1: {(end_1 - start):.4f} seconds\n')

    print('\n----------- SPLITTING INTO TRAINING AND TEST SET -----------\n')

    X_train, y_train, X_test, y_test = train_test_split(M, labels)

    print(f'Size of training set: {X_train.shape[0]}')
    print(f'Size of test set: {X_test.shape[0]}')

    end_2 = time.time()
    print(f'\nTime taken for splitting the dataset: {(end_2 - end_1):.4f} seconds\n')

    # Solve Part 2
    print('\n----------- PART 2 - NAIVE BAYES CLASSIFIER (WITHOUT LAPLACE CORRECTION) -----------\n')
    
    # Naive Bayes Classifier model without Laplace Correction
    NB = NaiveBayes(alpha=0)
    NB.fit(X_train, y_train, label_map)
    preds, _ = NB.predict(X_test)
    display_metrics(preds, y_test, label_map)

    end_3 = time.time()
    print(f'\nTime taken for Part 2: {(end_3 - end_2):.4f} seconds\n')

    # Solve Part 3
    print('\n----------- PART 3 - NAIVE BAYES CLASSIFIER WITH LAPLACE CORRECTION -----------\n')

    # Naive Bayes Classifier model with Laplace Correction
    NB_laplace = NaiveBayes(alpha=1)
    NB_laplace.fit(X_train, y_train, label_map)
    preds, _ = NB_laplace.predict(X_test)
    display_metrics(preds, y_test, label_map)

    end_4 = time.time()
    print(f'\nTime taken for Part 3: {(end_4 - end_3):.4f} seconds\n')
    
    print(f'Total time elapsed: {(end_4 - start):.4f} seconds\n')


if __name__ == '__main__':
    main()
