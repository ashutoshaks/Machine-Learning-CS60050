from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def principal_component_analysis(X_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray,
                                n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_valid_pca, X_test_pca
