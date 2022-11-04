from sklearn.decomposition import PCA
import numpy as np
import pathlib

IMG_PATH = str(pathlib.Path(__file__).parent.resolve() / 'images') + "/"


def prep_mnist(mnist_set, n_components=None, pca=None, apply_pca=True):
    filtered_mnist_set = []
    for image in mnist_set:
        if image[1] == 0 or image[1] == 1:
            filtered_mnist_set.append(image)

    X = [np.asarray(val[0]).flatten() for val in filtered_mnist_set]
    Y = [val[1] for val in filtered_mnist_set]

    if apply_pca:
        if not pca:
            pca = PCA(n_components=n_components)
            X_PC = pca.fit_transform(X)
        else:
            X_PC = pca.transform(X)
        return (X_PC, Y, pca)
    else:
        return (np.asarray(X), np.asarray(Y), pca)


def confusion_matrix(Y, Y_hat):
    # Y is the ground truth
    # Y_hat is the predicted labels
    # returns the confusion matrix
    cm = np.zeros((2, 2))
    for i in range(len(Y)):
        cm[Y[i], Y_hat[i]] += 1
    cm = cm.astype(int)
    return cm


def compute_accuracy(Y, Y_hat):
    # Y is the ground truth
    # Y_hat is the predicted labels
    # returns the accuracy
    cm = confusion_matrix(Y, Y_hat)
    return (cm[0, 0] + cm[1, 1]) / np.sum(cm)

def compute_error(Y, Y_hat):
    # Y is the ground truth
    # Y_hat is the predicted labels
    # returns the error
    cm = confusion_matrix(Y, Y_hat)
    return (cm[0, 1] + cm[1, 0]) / np.sum(cm)
