from data_preprocessing import process_data, train_test_split
from naive_bayes import NaiveBayes
from metrics import accuracy, ci
import time

def main():
    start = time.time()

    M, labels, vocab_map, label_map = process_data('train.csv')
    # print(M.shape)
    # print(labels.shape)
    # print(len(vocab_map))
    # print(label_map)

    X_train, y_train, X_test, y_test = train_test_split(M, labels)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    
    NB = NaiveBayes(alpha=1)
    NB.fit(X_train, y_train, label_map)
    print(NB.priors)
    print(NB.likelihoods)

    print()

    preds = NB.predict(X_test)
    acc = accuracy(y_test, preds)
    print(acc)
    print(ci(acc, y_train.shape[0]))
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
