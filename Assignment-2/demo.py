import pandas as pd
import numpy as np
import time
import re
from nltk.corpus import stopwords
from scipy.sparse import csr, lil_matrix, csr_matrix, vstack

if __name__ == '__main__':
    start = time.time()
    sw = set(stopwords.words("english"))

    df = pd.read_csv('train.csv')

    filtered_list = []
    words = []

    text = []
    for line in df['text']:
        tokens = re.findall("[a-z0-9]+", line.lower())
        text.append(tokens)

    for tokens in text:
        filtered = [w for w in tokens if not w in sw]
        filtered_list.append(filtered)
        words.extend(filtered)

    unique_words = sorted(list(set(words)))
    print(len(unique_words))

    # lim = 100
    # M = csr_matrix(np.zeros(len(unique_words)), dtype=bool)
    # flag = 1

    # for i in range(lim):
    #     line_set = set(text[i])
    #     Mnew = np.zeros(len(unique_words))
    #     for j, word in enumerate(unique_words):
    #         if word in line_set:
    #             Mnew[j] += 1
    #     if flag:
    #         M = csr_matrix(Mnew);
    #         flag = 0
    #     else:
    #         temp = csr_matrix(Mnew);
    #         M = vstack([M, Mnew])
    # print(M)

    ind = dict()
    for i, word in enumerate(unique_words):
        ind[word] = i


    lim = len(text)
    M = np.zeros(shape=(lim, len(unique_words)), dtype=int)


    # for i in range(lim):
    #     line_set = set(text[i])
    #     Mnew = np.zeros(len(unique_words))
    #     for j, word in enumerate(unique_words):
    #         if word in line_set:
    #             M[i][j] = 1

    for i, line in enumerate(filtered_list):
        for w in line:
            M[i][ind[w]] = 1

    print(M.shape)
    print((M == 1).sum())
    # print((M > 1).sum())
    # print(M.sum())

    # print(np.bincount(np.count_nonzero(M == True, axis=0)))

    # print(csr_matrix(M))
    # print(M);

    # A1 = np.array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1]])
    # A2 = np.array([[0, 0, 0, 1, 0, 0]])
    # print(A1)
    # print(A2)
    # A1sparse = csr_matrix(A1)
    # A2sparse = csr_matrix(A2)

    # C = vstack([A1sparse, A2sparse])
    # print(C)
    # print(C.toarray())

    end = time.time()
    print(end - start)

    arr = np.array(['a', 'b', 'a', 'c'])
    print(arr == 'a')
    print(type(arr))



# M - num_train x vocab_size

# class probabilities
# P(MWS), P(HPL), P(EAP) - c - np.array, c[0], c[1], c[2]

# P(w_i | author)
# np.array - 3 x vocab_size

# P(xi, xj, ... , xk | author)
