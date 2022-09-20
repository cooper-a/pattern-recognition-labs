import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA

def prep_mnist(mnist_set, n_components):
    # should we normalize the data?
    filtered_mnist_set = []
    for image in mnist_set:
        if image[1] == 0 or image[1] == 1:
            filtered_mnist_set.append(image)

    X = [np.asarray(val[0]).flatten() for val in filtered_mnist_set]
    Y = [val[1] for val in filtered_mnist_set]

    pca = PCA(n_components=n_components)
    X_PC = pca.fit_transform(X)
    return (X_PC, Y)


def k_nn(X, Y, x, k):
    # Implement the k-NN classifier
    # X is a list of vectors
    # Y is a list of labels
    # x is the vector to classify
    # k is the number of neighbors to consider
    # returns the most common class among the k nearest neighbors
    dist = []
    for i in range(len(X)):
        dist.append((np.linalg.norm(X[i] - x), Y[i]))
    # look at the k nearest neighbors
    dist.sort(key=lambda tup: tup[0])
    dist = dist[:k]
    # count the number of 0s and 1s
    count0 = 0
    count1 = 0
    for i in range(len(dist)):
        if dist[i][1] == 0:
            count0 += 1
        else:
            count1 += 1
    # return the most common class
    if count0 > count1:
        return 0
    else:
        return 1


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

X_PC, Y = prep_mnist(mnist_trainset)
X_test_PC, Y_test = prep_mnist(mnist_testset)

