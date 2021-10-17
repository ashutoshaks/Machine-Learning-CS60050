from data_processing import process_data, train_test_split
from naive_bayes import NaiveBayes
from metrics import accuracy, display_metrics
import time

def main():
    start = time.time()

    print('\n----------- PART 1 - LOADING AND PROCESSING DATA -----------\n')

    M, labels, vocab_map, label_map = process_data('train.csv')

    print(f'Shape of the binary feature matrix M: {M.shape}')
    print(f'Total number of examples: {M.shape[0]}')
    print(f'Vocabulary size: {M.shape[1]}')

    print('\n----------- SPLITTING INTO TRAINING AND TEST SET -----------\n')

    X_train, y_train, X_test, y_test = train_test_split(M, labels)

    print(f'Size of training set: {X_train.shape[0]}')
    print(f'Size of test set: {X_test.shape[0]}')

    print('\n----------- PART 2 - NAIVE BAYES CLASSIFIER (WITHOUT LAPLACE CORRECTION) -----------\n')
    
    NB = NaiveBayes(alpha=0)
    NB.fit(X_train, y_train, label_map)
    preds, _ = NB.predict(X_test)
    display_metrics(preds, y_test, label_map)

    print('\n----------- PART 3 - NAIVE BAYES CLASSIFIER WITH LAPLACE CORRECTION -----------\n')

    NB_laplace = NaiveBayes(alpha=1)
    NB_laplace.fit(X_train, y_train, label_map)
    preds, _ = NB_laplace.predict(X_test)
    display_metrics(preds, y_test, label_map)
    
    end = time.time()
    print(f'\nTotal time elapsed: {(end - start):.4f} seconds\n')


if __name__ == '__main__':
    main()
