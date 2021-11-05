from typing import Dict, Tuple
import numpy as np
from sklearn.svm import SVC
from multiprocessing import Pool

cnt = 0

def get_valid_score(svm: SVC) -> float:
    global cnt
    svm.fit(train_X, train_y)
    score = svm.score(valid_X, valid_y)
    print(svm, score, cnt)
    print()
    cnt += 1
    return score

def update_best(svm: SVC, score: float, best_score: float, best_svm: SVC):
    if score > best_score:
        best_score = score
        best_svm = svm
    return best_score, best_svm


def choose_best_SVM(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> Tuple[SVC, float, Dict]:
    global train_X, train_y, valid_X, valid_y
    train_X, train_y, valid_X, valid_y = X_train, y_train, X_valid, y_valid

    best_score = 0
    best_svm = None
    valid_acc = dict()

    svms = []

    for C in [0.01, 0.1, 1, 10, 100, 1000]:
        # linear kernel
        svm = SVC(C=C, kernel='linear')
        svms.append(svm)

        for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
            # rbf and sigmoid kernel
            for kernel in ['rbf', 'sigmoid']:
                svm = SVC(C=C, kernel=kernel, gamma=gamma)
                svms.append(svm)

            for degree in [2, 3, 4]:
                svm = SVC(C=C, kernel='poly', gamma=gamma, degree=degree)
                svms.append(svm)

    results = []
    with Pool(6) as p:
        results = p.map(get_valid_score, svms)

    i = np.argmax(results)
    return svms[i], results[i], svms, results


    # for C in [0.01, 0.1, 1, 10, 100, 1000]:
    #     valid_acc[C] = dict()

    #     # linear kernel
    #     svm = SVC(C=C, kernel='linear')
    #     score = get_valid_score(svm)
    #     valid_acc[C]['linear'] = score
    #     best_score, best_svm = update_best(svm, score, best_score, best_svm)

    #     for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
    #         valid_acc[C][gamma] = dict()

    #         # rbf and sigmoid kernel
    #         for kernel in ['rbf', 'sigmoid']:
    #             svm = SVC(C=C, kernel=kernel, gamma=gamma)
    #             score = get_valid_score(svm)
    #             valid_acc[C][gamma][kernel] = score
    #             best_score, best_svm = update_best(svm, score, best_score, best_svm)

    #         # poly kernel
    #         for degree in [2, 3, 4]:
    #             valid_acc[C][gamma][degree] = dict()
    #             svm = SVC(C=C, kernel='poly', gamma=gamma, degree=degree)
    #             score = get_valid_score(svm)
    #             valid_acc[C][gamma][degree] = score
    #             best_score, best_svm = update_best(svm, score, best_score, best_svm)
    # return best_svm, best_score, valid_acc
