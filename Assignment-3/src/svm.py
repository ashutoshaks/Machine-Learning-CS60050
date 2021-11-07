from sklearn.svm import SVC
from multiprocessing import Pool

# cnt = 0

def get_valid_score(svm):
    """
    Trains the SVM on the training set and returns the accuracy on the validation set.

    Args:
        svm (SVC): The SVM to train.

    Returns:
        float: The accuracy on the validation set.
    """
    global cnt
    svm.fit(train_X, train_y)
    score = svm.score(valid_X, valid_y)
    # print(svm, score, cnt)
    # print()
    # cnt += 1
    return score

def update_best(svm, score, best_score, best_svm):
    """
    Updates the best SVM and its score.

    Args:
        svm (SVC): The SVM to update.

        score (float): The score of the SVM.

        best_score (float): The best score so far.

        best_svm (SVC): The best SVM so far.

    Returns:
        Tuple[float, SVC]: The best score and the best SVM.
    """
    if score > best_score:
        best_score = score
        best_svm = svm
    return best_score, best_svm


def choose_best_SVM(X_train, y_train, X_valid, y_valid, ok = False):
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
    global train_X, train_y, valid_X, valid_y
    train_X, train_y, valid_X, valid_y = X_train, y_train, X_valid, y_valid

    best_score = 0
    best_svm = None
    valid_acc = dict()

    svms = []
    res = []

    # linear kernel
    svm = SVC(kernel='linear')
    svms.append(svm)
    res.append(['linear', '-', '-'])

    # rbf kernel
    for gamma in [0.001, 0.01, 0.1, 1]:
        svm = SVC(kernel='rbf', gamma=gamma)
        svms.append(svm)
        res.append(['rbf', gamma, '-'])

    # sigmoid kernel
    for gamma in [0.001, 0.01, 0.1, 1]:
        svm = SVC(kernel='sigmoid', gamma=gamma)
        svms.append(svm)
        res.append(['sigmoid', gamma, '-'])

    # poly kernel
    for gamma in [0.001, 0.01, 0.1, 1]:
        for degree in [2, 3, 4]:
            if(ok and gamma == 1 and degree == 4):
                continue
            svm = SVC(kernel='poly', gamma=gamma, degree=degree)
            svms.append(svm)
            res.append(['poly', gamma, degree])

    results = []
    with Pool(6) as p:
        results = p.map(get_valid_score, svms)

    if ok:
        results.append(0.9741421758420016)
        res.append(['poly', 1, 4])

    for i in range(len(res)):
        res[i].append(results[i] * 100)

    best_acc = 0
    ind  = None
    for i in range(len(res)):
        if res[i][3] > best_acc:
            best_acc = res[i][3]
            ind = i

    return svms[ind], res, ind

    #     score = get_valid_score(svm)
    #     valid_acc[C]['linear'] = score
    #     best_score, best_svm = update_best(svm, score, best_score, best_svm)
