from typing import Dict, List, Tuple
import numpy as np


class NaiveBayes:
    """
    The NaiveBayes class having metadata for the naive Bayes classifier, and functions
    for various operations of the classifier.
    """

    def __init__(self, alpha: int = 0) -> None:
        """
        Initializes the naive Bayes classifier with proper metadata.

        Args:
            alpha (int, optional): The indicator for whether we want Laplace correction or not.
                alpha = 0 corresponds to no Laplace correction, and alpha = 0 corresponds to 
                Laplace correction. Defaults to 0.
        """
        self.alpha = alpha


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, label_map: Dict[int, str]) -> None:
        """
        Initializes other attributes for the naive Bayes classifier, and calls the train method.

        Args:
            X_train (np.ndarray): The feature matrix corresponding to the training data.

            y_train (np.ndarray): The labels corresponding to the training data.

            label_map (Dict[int, str]): A dictionary mapping the integer labels to the author names.
        """
        self.num_features = X_train.shape[1]
        self.num_classes = len(label_map)
        self.label_map = label_map
        self.priors = np.empty(shape=self.num_classes)
        self.likelihoods = np.empty(shape=(self.num_classes, self.num_features))

        self.train(X_train, y_train, label_map)


    def train(self, train_features: np.ndarray, labels: np.ndarray, label_map: Dict[int, str]) -> None:
        """
        Trains the naive Bayes classifier.

        Args:
            train_features (np.ndarray): The feature matrix corresponding to the training data.

            labels (np.ndarray): The labels corresponding to the training data.

            label_map (Dict[int, str]): A dictionary mapping the integer labels to the author names.
        """
        num_examples = labels.shape[0]
        
        for i in range(self.num_classes):
            mask = (labels == i)
            count_i = np.sum(mask)

            # calculate priors
            self.priors[i] = count_i / num_examples

            features_i = train_features[mask]
            occ_cnt = np.sum(features_i, axis = 0)

            # calculate likelihoods
            self.likelihoods[i] = (occ_cnt + self.alpha) / (features_i.shape[0] + self.alpha * self.num_features)


    def predict_one(self, features: np.ndarray) -> Tuple[int, str]:
        """
        Predicts the classisication of a single sample on the basis of the naive Bayes classifier created.

        Args:
            features (np.ndarray): The feature vector for the test example.

        Returns:
            Tuple[int, str]: The label predicted, and the corresponding author name.
        """
        probs = np.prod(self.likelihoods[:, features == 1], axis=1) * self.priors
        label = np.argmax(probs)
        return (label, self.label_map[label])


    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Predicts the classification on a set of test data.

        Args:
            test_features (np.ndarray): The feature matrix for the test data.

        Returns:
            Tuple[np.ndarray, List[str]]: The array of labels predicted, and a list of corresponding author names.
        """
        preds = [self.predict_one(feature) for feature in test_features]
        preds_label, preds_str = zip(*preds)
        return (np.array(preds_label), list(preds_str))
