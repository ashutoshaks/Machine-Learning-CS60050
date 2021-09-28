import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

import time

def process_data(file: str):
    df = pd.read_csv(file)

    text_list = []
    vocabulary = []

    stop_words = set(stopwords.words("english"))

    for line in df['text']:
        tokens = re.findall("[a-z0-9]+", line.lower())
        filtered_line = [word for word in tokens if not word in stop_words]
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

    author_map = {'EAP': 0,'HPL': 1,'MWS': 2}
    labels = df['author'].map(author_map).to_numpy()

    return (M, labels, vocab_map, author_map)


def train_test_split(features: np.ndarray, labels: np.ndarray, train_ratio: float = 0.7):
    # idx = np.random.permutation(len(features))
    # features, labels = features[idx], labels[idx]

    split_index = int(train_ratio * len(features))
    X_train, y_train = features[:split_index], labels[:split_index]
    X_test, y_test = features[split_index:], labels[split_index:]

    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    start = time.time()

    M, labels, vocab_map, author_map = process_data('train.csv')
    print(M.shape)
    print(labels.shape)
    print(len(vocab_map))
    print(author_map)

    print()

    X_train, y_train, X_test, y_test = train_test_split(M, labels)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    end = time.time()
    print(end - start)
