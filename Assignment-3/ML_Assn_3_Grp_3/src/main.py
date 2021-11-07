# Machine Learning - Assignment 3
# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008

from data_processing import read_data, split_data, normalize
from feature_extraction import perform_PCA, perform_LDA, plot_LDA, plot_PCA, scree_plot_pca
from metrics import display_metrics
from svm import choose_best_SVM
from tabulate import tabulate

def main():
    print('\n---------------- PART 1 - LOADING, SPLITTING AND PROCESSING DATA ----------------\n')
    
    X, y = read_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y)
    X_train, X_valid, X_test = normalize(X_train, X_valid, X_test)
    print(f'Shape of X_train: {X_train.shape}')
    print(f'Shape of X_valid: {X_valid.shape}')
    print(f'Shape of X_test: {X_test.shape}')

    print('\n---------------- PART 2 - PERFORMING PRINCIPAL COMPONENT ANALYSIS ----------------\n')
    
    X_train_pca, X_valid_pca, X_test_pca, pca = perform_PCA(X_train, X_valid, X_test)
    print(f'Shape of X_train_pca: {X_train_pca.shape}')
    print(f'Shape of X_valid_pca: {X_valid_pca.shape}')
    print(f'Shape of X_test_pca: {X_test_pca.shape}')
    plot_PCA(X_train_pca, y_train)
    scree_plot_pca(pca)

    print('\n---------------- PART 3 - FINDING BEST SVM AFTER PCA ----------------\n')
    best_svm, results, ind = choose_best_SVM(X_train_pca, y_train, X_valid_pca, y_valid)

    print(tabulate(results, headers=['kernel', 'gamma', 'degree', 'accuracy'], floatfmt=".4f", tablefmt='fancy_grid'))
    print('\nKernel parameters for highest validation set accuracy:')
    print(f'kernel: {results[ind][0]}, gamma: {results[ind][1]}, degree: {results[ind][2]}')
    print(f'Validation set accuracy: {results[ind][3]:.4f}')
    best_svm.fit(X_train_pca, y_train)
    preds = best_svm.predict(X_test_pca)
    score = best_svm.score(X_test_pca, y_test) * 100
    print(f'Test set accuracy: {score:.4f}')
    print('\nClassification metrics for best SVM after PCA')
    display_metrics(preds, y_test)

    print('\n---------------- PART 4 - PERFORMING LINEAR DISCRIMINANT ANALYSIS ----------------\n')
    
    X_train_lda, X_valid_lda, X_test_lda, lda = perform_LDA(X_train, y_train, X_valid, X_test)
    print(f'Shape of X_train_lda: {X_train_lda.shape}')
    print(f'Shape of X_valid_lda: {X_valid_lda.shape}')
    print(f'Shape of X_test_lda: {X_test_lda.shape}')    
    plot_LDA(X_train_lda, y_train)

    print('\n---------------- PART 5 - FINDING BEST SVM AFTER LDA ----------------\n')
    best_svm, results, ind = choose_best_SVM(X_train_lda, y_train, X_valid_lda, y_valid, True)

    print(tabulate(results, headers=['kernel', 'gamma', 'degree', 'accuracy'], floatfmt=".4f", tablefmt='fancy_grid'))
    print('\nKernel parameters for highest validation set accuracy:')
    print(f'kernel: {results[ind][0]}, gamma: {results[ind][1]}, degree: {results[ind][2]}')
    print(f'Validation set accuracy: {results[ind][3]:.4f}')
    best_svm.fit(X_train_lda, y_train)
    preds = best_svm.predict(X_test_lda)
    score = best_svm.score(X_test_lda, y_test) * 100
    print(f'Test set accuracy: {score:.4f}')
    print('\nClassification metrics for best SVM after LDA:')
    display_metrics(preds, y_test)


if __name__ == "__main__":
    main()
