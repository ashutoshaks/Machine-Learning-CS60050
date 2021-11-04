import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.array([[5, 3, 2], [4, 6, 0], [3, -7, 14]])
X1 = np.array([[5, 3, 2]])
pca = PCA(n_components=2)
sc = StandardScaler(with_mean=False)
sc.fit(X)
X = sc.transform(X)
print(X)


pca.fit(X)
B = pca.components_
print(pca.transform(X))

# X1 = sc.fit_transform(X)
# print(X1)
# X1 = X1 - mean
# print(X1)
# X1 = (X1) / np.sqrt(sc.var_)
X1 = sc.transform(X1)
print(pca.transform(X1))
# print(X1 @ B.T)

# print(pca.transform(X1))

# X1s = sc.fit_transform(X1)
# print(X1s @ pca.components_.T)

# X1s = sc.fit_transform(X1)

# print(pca.transform(X1s))

# B = pca.components_
# print(B)

# print(pca.fit_transform(X))
# print(X @ B.T)
