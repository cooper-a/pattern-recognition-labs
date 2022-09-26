from sklearn.decomposition import PCA
import numpy as np


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

def confusion_matrix(Y, Y_hat):
    # Y is the ground truth
    # Y_hat is the predicted labels
    # returns the confusion matrix
    cm = np.zeros((2, 2))
    for i in range(len(Y)):
        cm[Y[i], Y_hat[i]] += 1
    return cm
