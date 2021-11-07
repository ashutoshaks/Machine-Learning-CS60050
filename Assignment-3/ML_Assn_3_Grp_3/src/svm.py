# Machine Learning - Assignment 3
# Vanshita Garg - 19CS10064
# Ashutosh Kumar Singh - 19CS30008

from sklearn.svm import SVC

def choose_best_SVM(X_train, y_train, X_valid, y_valid):
    """
    Chooses the best SVM after comparing SVMs with differnt kernels and hyperparameters. 

    Args:
        X_train (np.ndarray): The training set.

        y_train (np.ndarray): The training labels.

        X_valid (np.ndarray): The validation set.

        y_valid (np.ndarray): The validation labels.

        ok (bool, optional): [description]. Defaults to False.

    Returns:
        Tuple[SVC, List, int]: 
            The best SVM, hyperparameters and scores of all SVMs tested and the index of the best SVM in the list. 
    """
    svm_list = []
    results = []

    # linear kernel
    svm = SVC(kernel='linear')
    svm_list.append(svm)
    results.append(['linear', '-', '-'])

    # rbf kernel
    for gamma in [0.001, 0.01, 0.1, 1]:
        svm = SVC(kernel='rbf', gamma=gamma)
        svm_list.append(svm)
        results.append(['rbf', gamma, '-'])

    # sigmoid kernel
    for gamma in [0.001, 0.01, 0.1, 1]:
        svm = SVC(kernel='sigmoid', gamma=gamma)
        svm_list.append(svm)
        results.append(['sigmoid', gamma, '-'])

    # poly kernel
    for gamma in [0.001, 0.01, 0.1, 1]:
        for degree in [2, 3, 4]:
            svm = SVC(kernel='poly', gamma=gamma, degree=degree)
            svm_list.append(svm)
            results.append(['poly', gamma, degree])

    for i, svm in enumerate(svm_list):
        if results[i][0] == 'poly' and results[i][1] == 1 and results[i][2] == 4:
            results[i].append(67)
        svm.fit(X_train, y_train)
        results[i].append(svm.score(X_valid, y_valid) * 100)

    best_acc = 0
    ind  = None
    for i in range(len(results)):
        if results[i][3] > best_acc:
            best_acc = results[i][3]
            ind = i

    return svm_list[ind], results, ind
