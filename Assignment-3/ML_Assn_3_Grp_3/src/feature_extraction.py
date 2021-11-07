# Machine Learning - Assignment 3
# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt

def perform_PCA(X_train, X_valid, X_test, n_components = 2):
    """
    Performs PCA on the training set and then applies the same 
    transformation vector on the validation and test set data.

    Args:
        X_train (np.ndarray): Training data

        X_valid (np.ndarray): Validation data

        X_test (np.ndarray): Test data

        n_components (int, optional): Number of components to keep. Defaults to 2.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]: Transformed training, validation, and test data, and the PCA object
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_valid_pca, X_test_pca, pca


def perform_LDA(X_train, y_train, X_valid, X_test, n_components: int = 1):
    """
    Performs LDA on the training set and then applies the same
    transformation vector on the validation and test set data.

    Args:
        X_train (np.ndarray): Training data

        y_train (np.ndarray): Training labels

        X_valid (np.ndarray): Validation data

        X_test (np.ndarray): Test data

        n_components (int, optional): Number of components to keep. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, LinearDiscriminantAnalysis]: Transformed training, validation, and test data, and the LDA object
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    X_valid_lda = lda.transform(X_valid)
    X_test_lda = lda.transform(X_test)

    return X_train_lda, X_valid_lda, X_test_lda, lda


def plot_PCA(X, y):
    """
    Plots the two principal components of the training data after PCA.

    Args:
        X (np.ndarray): 2-D Training data after having performed PCA

        y (np.ndarray): Training labels
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap='coolwarm')
    plt.title('Training Data after PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('../plots/pca.png')
    plt.clf()


def scree_plot_pca(pca):
    """
    Plots the scree plot of the PCA object.

    Args:
        pca (PCA): PCA object
    """
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    print('\nPercentage of Explained Variance:')
    print(f'PC1: {per_var[0]}%\nPC2: {per_var[1]}%')
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, width=0.6, tick_label=labels)
    for i, v in enumerate(per_var):
        plt.text(x = i + 0.9, y = v + 0.7, s = str(v) + '%', size = 10)
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage of Variance')
    plt.title('Scree Plot')
    plt.savefig('../plots/pca_scree.png')
    plt.clf()

def plot_LDA(X, y): 
    """
    Plots the training data after LDA.

    Args:
        X (np.ndarray): 1-D Training data after having performed LDA
        
        y (np.ndarray): Training labels
    """
    plt.scatter(X, np.ones(shape=X.shape[0]), c=y, s=30, cmap='coolwarm')
    plt.title('Training Data after LDA')
    plt.xlabel('X')
    plt.yticks([])
    plt.savefig('../plots/lda.png')
    plt.clf()
