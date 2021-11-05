from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt

def perform_PCA(X_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray,
                                n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_valid_pca, X_test_pca


def perform_LDA(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray,
                                n_components: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    X_valid_lda = lda.transform(X_valid)
    X_test_lda = lda.transform(X_test)

    return X_train_lda, X_valid_lda, X_test_lda


def plot_PCA(X: np.ndarray, y: np.ndarray) -> None:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap='coolwarm')
    plt.title('Training Data after PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('plots/pca.png')
    plt.show()
    plt.clf()


def plot_LDA(X: np.ndarray, y: np.ndarray) -> None:
    plt.scatter(X, np.ones(shape=X.shape[0]), c=y, s=30, cmap='coolwarm')
    plt.title('Training Data after LDA')
    plt.xlabel('X')
    plt.yticks([])
    plt.savefig('plots/lda.png')
    plt.show()
