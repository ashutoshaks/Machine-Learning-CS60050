import random
from data_processing import read_data, split_data, normalize
from feature_extraction import perform_PCA, perform_LDA, plot_LDA, plot_PCA
from metrics import display_metrics
from svm import choose_best_SVM
from sklearn.svm import SVC
import numpy as np

def main():
    # random.seed(0)
    # np.random.seed(0)
    
    X, y = read_data()
    # print(np.bincount(y))
    # print(X[:, 2])
    # print(np.mean(y == (X[:, 2] > 0)))
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y)
    X_train, X_valid, X_test = normalize(X_train, X_valid, X_test)
    X_train_pca, X_valid_pca, X_test_pca = perform_PCA(X_train, X_valid, X_test)
    svm = SVC()
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    # display_metrics(preds, y_test)

    # plot_PCA(X_train_pca, y_train)
    # best_svm_pca, best_score_pca, svms_pca, results_pca = choose_best_SVM(X_train_pca, y_train, X_valid_pca, y_valid)

    # print(best_svm_pca)
    # print(best_score_pca)
    # best_svm_pca.fit(X_train_pca, y_train)
    # print(best_svm.score(X_test_pca, y_test))
    # print(results_pca)

    # X_train_lda, X_valid_lda, X_test_lda = perform_LDA(X_train, y_train, X_valid, X_test)
    # plot_LDA(X_train_lda, y_train)
    # best_svm_lda, best_score_lda, svms_lda, results_lda = choose_best_SVM(X_train_lda, y_train, X_valid_lda, y_valid)
    # print(best_svm_lda)
    # print(best_score_lda)
    # best_svm_lda.fit(X_train_lda, y_train)
    # print(best_svm_lda.score(X_test_lda, y_test))
    # print(results_lda)


if __name__ == "__main__":
    main()
