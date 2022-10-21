import torchvision.datasets as datasets
import numpy as np
from a2_utils import prep_mnist, confusion_matrix, compute_accuracy, compute_error, IMG_PATH
import matplotlib.pyplot as plt
from pathlib import Path

class MLE_Exponential_Classifier:
    # Implement the MLE classifier assuming a 1 dimensional exponential distribution
    def __init__(self, X, Y):
        self.X_clf = X
        self.Y_clf = Y
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]

        # filter out negative values
        X0 = [x for x in X0 if x[0] > 0]
        X1 = [x for x in X1 if x[0] > 0]

        self.mean0 = np.mean(X0, axis=0)
        self.mean1 = np.mean(X1, axis=0)
        self.lambda0 = 1 / self.mean0
        self.lambda1 = 1 / self.mean1

    def classify(self, x):
        self.__check_if_clf_trained()
        if self.lambda0 * np.exp(-self.lambda0 * x) > self.lambda1 * np.exp(-self.lambda1 * x):
            return 0
        else:
            return 1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.mean0 is None or self.mean1 is None:
            raise Exception("Classifier is not trained yet.")


class MLE_Uniform_Classifier:
    # Implement the MLE classifier assuming a 1 dimensional uniform distribution
    def __init__(self, X, Y):
        self.X_clf = X
        self.Y_clf = Y
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]

        self.mean0 = np.mean(X0, axis=0)
        self.mean1 = np.mean(X1, axis=0)

        self.a0 = min(X0)[0]
        self.a1 = min(X1)[0]
        self.b0 = max(X0)[0]
        self.b1 = max(X1)[0]

    def classify(self, x):
        self.__check_if_clf_trained()
        if self.a0 < x < self.b0:
            prob0 = 1 / (self.b0 - self.a0)
        else:
            prob0 = 0
        if self.a1 < x < self.b1:
            prob1 = 1 / (self.b1 - self.a1)
        else:
            prob1 = 0
        if prob0 > prob1:
            return 0
        else:
            return 1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.mean0 is None or self.mean1 is None:
            raise Exception("Classifier is not trained yet.")


class MLE_Gaussian_Classifier:
    # Implement the MLE classifier assuming a 1 dimensional gaussian distribution
    def __init__(self, X, Y):
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        self.X_clf = X
        self.Y_clf = Y
        self.mean0 = np.mean(X0, axis=0)
        self.mean1 = np.mean(X1, axis=0)
        self.var0 = np.var(X0, axis=0)
        self.var1 = np.var(X1, axis=0)

    def classify(self, x):
        log_p0 = -np.log(np.sqrt(2 * np.pi * self.var0)) - (x - self.mean0) ** 2 / (2 * self.var0)
        log_p1 = -np.log(np.sqrt(2 * np.pi * self.var1)) - (x - self.mean1) ** 2 / (2 * self.var1)
        if log_p0 > log_p1:
            return 0
        else:
            return 1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.mean0 is None or self.mean1 is None:
            raise Exception("Classifier is not trained yet.")



def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    X_PC, Y, pca = prep_mnist(mnist_trainset, 1)
    X_test_PC, Y_test, _ = prep_mnist(mnist_testset, 1, pca)

    mle_exp_clf = MLE_Exponential_Classifier(X_PC, Y)
    mle_uni_clf = MLE_Uniform_Classifier(X_PC, Y)
    mle_gauss_clf = MLE_Gaussian_Classifier(X_PC, Y)

    Y_pred_exp = mle_exp_clf.predict(X_test_PC)
    Y_pred_uni = mle_uni_clf.predict(X_test_PC)
    Y_pred_gauss = mle_gauss_clf.predict(X_test_PC)

    conf_mat_exp = confusion_matrix(Y_test, Y_pred_exp)
    conf_mat_uni = confusion_matrix(Y_test, Y_pred_uni)
    conf_mat_gauss = confusion_matrix(Y_test, Y_pred_gauss)

    # print("Confusion Matrix for MLE Exponential Classifier")
    # print(conf_mat_exp)
    # print("Confusion Matrix for MLE Uniform Classifier")
    # print(conf_mat_uni)
    # print("Confusion Matrix for MLE Gaussian Classifier")
    # print(conf_mat_gauss)
    #
    # print("Accuracy for MLE Exponential Classifier")
    # print(compute_accuracy(Y_test, Y_pred_exp))
    # print("Accuracy for MLE Uniform Classifier")
    # print(compute_accuracy(Y_test, Y_pred_uni))
    # print("Accuracy for MLE Gaussian Classifier")
    # print(compute_accuracy(Y_test, Y_pred_gauss))

    print("Error for MLE Exponential Classifier is: ", round(compute_error(Y_test, Y_pred_exp), 7))
    print("The lambda0 is: ", mle_exp_clf.lambda0)
    print("The lambda1 is: ", mle_exp_clf.lambda1)

    # print a line to separate the output of different classifiers
    print("--------------------------------------------------")
    print("Error for MLE Uniform Classifier is: ", round(compute_error(Y_test, Y_pred_uni), 7))
    print("The a0 is: ", mle_uni_clf.a0)
    print("The a1 is: ", mle_uni_clf.a1)
    print("The b0 is: ", mle_uni_clf.b0)
    print("The b1 is: ", mle_uni_clf.b1)

    # print a line to separate the output of different classifiers
    print("--------------------------------------------------")
    print("Error for MLE Gaussian Classifier is: ", round(compute_error(Y_test, Y_pred_gauss), 7))
    print("The mean0 is: ", mle_gauss_clf.mean0)
    print("The mean1 is: ", mle_gauss_clf.mean1)
    print("The var0 is: ", mle_gauss_clf.var0)
    print("The var1 is: ", mle_gauss_clf.var1)


if __name__ == "__main__":
    main()
