# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008
# Machine Learning - Assignment 2

import pandas as pd
import numpy as np
import re

# The list of all English stopwords
stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", 
                "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", 
                "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
                "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
                "about", "against", "between", "into", "through", "during", "before", "after", 
                "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
                "under", "again", "further", "then", "once", "here", "there", "when", "where", 
                "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
                "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
                "very", "s", "t", "can", "will", "just", "don", "should", "now"])


def tokenize(line):
    """
    Tokenizes a string (sentence) into a list of words after removing the stopwords.

    Args:
        line (str): The sentence that needs to be tokenized.

    Returns:
        List[str]: The tokenized sentence.
    """
    tokens = re.findall('[a-z0-9]+', line.lower())
    filtered_line = [word for word in tokens if not word in stop_words]
    return filtered_line


def process_data(file):
    """
    Constructs the r × c binary feature matrix M where r is the number of examples and c is the 
    size of the vocabulary consisting of distinct words present in the dataset. Each row
    corresponds to an example of the dataset and each column corresponds to a word in the 
    vocabulary. M[i, j] = 1 if and only if the j-th word is present in the text of the i example.

    Args:
        file (str): The data file.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[int, str]]: The binary feature matrix M,
            the list of labels for each example, and a dictionary mapping the integer 
            labels to the author names.
    """
    df = pd.read_csv(file)

    text_list = []
    vocabulary = []

    for line in df['text']:
        filtered_line = tokenize(line)
        text_list.append(filtered_line)
        vocabulary.extend(filtered_line)

    # The vocabulary consisting of all words
    vocabulary = sorted(list(set(vocabulary)))

    vocab_map = dict()
    for i, word in enumerate(vocabulary):
        vocab_map[word] = i

    M = np.zeros(shape=(len(text_list), len(vocabulary)), dtype=bool)

    # Populate the binary feature matrix M
    for i, line in enumerate(text_list):
        for word in line:
            M[i, vocab_map[word]] = 1

    # {'EAP': 0, 'HPL': 1, 'MWS': 2}
    author_map = {author: i for i, author in enumerate(df['author'].unique())}
    # {0: 'EAP', 1: 'HPL', 2: 'MWS'}
    label_map = {i: author for author, i in author_map.items()}
    labels = df['author'].map(author_map).to_numpy()

    return (M, labels, label_map)


def train_test_split(features, labels, train_ratio = 0.7):
    """
    Shuffles the data and splits the dataset into a train-test split.

    Args:
        features (np.ndarray): The binary feature matrix.

        labels (np.ndarray): The labels corresponding to each example.

        train_ratio (float, optional): The fraction of examples in the train split. Defaults to 0.7.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Features for the training data, 
            labels of the training data, features for the test data, labels of the test data.
    """
    idx = np.random.permutation(len(features))
    features, labels = features[idx], labels[idx]

    split_index = int(train_ratio * len(features))
    X_train, y_train = features[:split_index], labels[:split_index]
    X_test, y_test = features[split_index:], labels[split_index:]

    return (X_train, y_train, X_test, y_test)
