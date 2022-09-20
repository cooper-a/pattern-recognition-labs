import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

filtered_mnist_train = []
for image in mnist_trainset:
    if image[1] == 0 or image[1] == 1:
        filtered_mnist_train.append(image)

X = [np.asarray(val[0]).flatten() for val in filtered_mnist_train]
Y = [val[1] for val in filtered_mnist_train]

pca = PCA(n_components=20)
X_PC = pca.fit_transform(X)


def minimum_euclidian_clf(X, x):
    # Implement the MED classifier
    # X is a list of vectors
    # x is the vector to classify
    # returns the most common class among the k nearest neighbors
    dist = []
    for i in range(len(X)):
        dist.append((np.linalg.norm(X[i] - x), Y[i]))
    # look for the closest neighbor
    dist.sort(key=lambda tup: tup[0])
    return dist[0][1]

