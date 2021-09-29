from typing import Dict, List, Tuple
import numpy as np

class NaiveBayes:

    def __init__(self, alpha: int = 0) -> None:
        self.alpha = alpha


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, label_map: Dict) -> None:
        self.num_features = X_train.shape[1]
        self.num_classes = len(label_map)
        self.label_map = label_map
        self.priors = np.empty(shape=self.num_classes)
        self.likelihoods = np.empty(shape=(self.num_classes, self.num_features))

        self.train(X_train, y_train, label_map)


    def train(self, train_features: np.ndarray, labels: np.ndarray, label_map: Dict) -> None:
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
        probs = np.prod(self.likelihoods[:, features == 1], axis=1) * self.priors
        label = np.argmax(probs)
        return (label, self.label_map[label])


    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        preds = [self.predict_one(feature) for feature in test_features]
        preds_label, preds_str = zip(*preds)
        return (np.array(preds_label), list(preds_str))
