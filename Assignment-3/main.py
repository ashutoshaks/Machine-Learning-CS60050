from data_processing import read_data, split_data, normalize
from feature_extraction import principal_component_analysis

def main():
    X, y = read_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y)
    X_train, X_valid, X_test = normalize(X_train, X_valid, X_test)
    X_train_pca, X_valid_pca, X_test_pca = principal_component_analysis(X_train, X_valid, X_test)
    

if __name__ == "__main__":
    main()
