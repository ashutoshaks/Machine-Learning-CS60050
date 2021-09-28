from metrics import accuracy
from naive_bayes import NaiveBayes
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

import time

stop_words = set(stopwords.words("english"))

def tokenize(line: str):
    tokens = re.findall("[a-z0-9]+", line.lower())
    filtered_line = [word for word in tokens if not word in stop_words]
    return filtered_line


def process_data(file: str):
    df = pd.read_csv(file)

    text_list = []
    vocabulary = []

    for line in df['text']:
        filtered_line = tokenize(line)
        text_list.append(filtered_line)
        vocabulary.extend(filtered_line)

    vocabulary = sorted(list(set(vocabulary)))

    vocab_map = dict()
    for i, word in enumerate(vocabulary):
        vocab_map[word] = i

    M = np.zeros(shape=(len(text_list), len(vocabulary)), dtype=int)

    for i, line in enumerate(text_list):
        for word in line:
            M[i][vocab_map[word]] = 1

    # author_map = {'EAP': 0,'HPL': 1,'MWS': 2}
    author_map = {author: i for i, author in enumerate(df['author'].unique())}
    label_map = {i: author for author, i in author_map.items()}
    labels = df['author'].map(author_map).to_numpy()

    return (M, labels, vocab_map, label_map)


def train_test_split(features: np.ndarray, labels: np.ndarray, train_ratio: float = 0.7):
    idx = np.random.permutation(len(features))
    features, labels = features[idx], labels[idx]

    split_index = int(train_ratio * len(features))
    X_train, y_train = features[:split_index], labels[:split_index]
    X_test, y_test = features[split_index:], labels[split_index:]

    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
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
    print(accuracy(y_test, preds))

    end = time.time()
    print(end - start)
