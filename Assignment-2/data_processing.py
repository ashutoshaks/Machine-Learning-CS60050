from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import re

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


def tokenize(line: str) -> List[str]:
    tokens = re.findall('[a-z0-9]+', line.lower())
    filtered_line = [word for word in tokens if not word in stop_words]
    return filtered_line


def process_data(file: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
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

    M = np.zeros(shape=(len(text_list), len(vocabulary)), dtype=bool)

    for i, line in enumerate(text_list):
        for word in line:
            M[i, vocab_map[word]] = 1

    # {'EAP': 0,'HPL': 1,'MWS': 2}
    author_map = {author: i for i, author in enumerate(df['author'].unique())}
    label_map = {i: author for author, i in author_map.items()}
    labels = df['author'].map(author_map).to_numpy()

    return (M, labels, vocab_map, label_map)


def train_test_split(features: np.ndarray, labels: np.ndarray, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.permutation(len(features))
    features, labels = features[idx], labels[idx]

    split_index = int(train_ratio * len(features))
    X_train, y_train = features[:split_index], labels[:split_index]
    X_test, y_test = features[split_index:], labels[split_index:]

    return (X_train, y_train, X_test, y_test)
