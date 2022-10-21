import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a2_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class MLE_Classifier:
    def __init__(self, X, Y):
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        # compute the mean of each class
        self.X_clf = X
        self.Y_clf = Y
        self.mean0 = np.mean(X0, axis=0)
        self.mean1 = np.mean(X1, axis=0)
        self.covariance0 = np.cov(X0, rowvar=False)
        self.covariance1 = np.cov(X1, rowvar=False)

    def classify(self, x):
        # Implement the MLE classifier
        # x is a single vector to classify
        # returns the most likely class
        # compute the log likelihood of each class
        log_likelihood0 = -0.5 * np.dot(np.dot((x - self.mean0).T, np.linalg.inv(self.covariance0)), (x - self.mean0)) - 0.5 * np.log(np.linalg.det(self.covariance0))
        log_likelihood1 = -0.5 * np.dot(np.dot((x - self.mean1).T, np.linalg.inv(self.covariance1)), (x - self.mean1)) - 0.5 * np.log(np.linalg.det(self.covariance1))
        # return the most likely class
        if log_likelihood0 > log_likelihood1:
            return 0
        else:
            return 1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.mean0 is None or self.mean1 is None:
            raise Exception("Classifier is not trained yet.")

    def plot_decision_boundary(self, h=10):
        self.__check_if_clf_trained()
        x_min, x_max = self.X_clf[:, 0].min() - 1, self.X_clf[:, 0].max() + 1
        y_min, y_max = self.X_clf[:, 1].min() - 1, self.X_clf[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                Z[i, j] = self.classify(np.array([xx[i, j], yy[i, j]]))
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(self.X_clf[:, 0], self.X_clf[:, 1], c=self.Y_clf, cmap=plt.cm.coolwarm)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        red_patch = mpatches.Patch(color='red', label='Class 0')
        blue_patch = mpatches.Patch(color='blue', label='Class 1')
        plt.legend(handles=[red_patch, blue_patch])
        plt.title('MLE Decision Boundary (2D)')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        path = IMG_PATH + f"MLE_Decision_Boundary_2D.png"
        plt.savefig(path)
        plt.show()


class MAP_Classifier:
    def __init__(self, X, Y):
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        # compute the mean of each class
        self.X_clf = X
        self.Y_clf = Y
        self.mean0 = np.mean(X0, axis=0)
        self.mean1 = np.mean(X1, axis=0)
        self.covariance0 = np.cov(X0, rowvar=False)
        self.covariance1 = np.cov(X1, rowvar=False)
        self.log_prior0 = np.log(len(X0) / len(X))
        self.log_prior1 = np.log(len(X1) / len(X))

    def classify(self, x):
        # Implement the MAP classifier
        # x is a single vector to classify
        # returns the most likely class
        # compute the log likelihood of each class
        log_likelihood0 = -0.5 * np.dot(np.dot((x - self.mean0).T, np.linalg.inv(self.covariance0)), (x - self.mean0)) - 0.5 * np.log(np.linalg.det(self.covariance0))
        log_likelihood1 = -0.5 * np.dot(np.dot((x - self.mean1).T, np.linalg.inv(self.covariance1)), (x - self.mean1)) - 0.5 * np.log(np.linalg.det(self.covariance1))
        # compute the log posterior of each class
        log_posterior0 = log_likelihood0 + self.log_prior0
        log_posterior1 = log_likelihood1 + self.log_prior1
        # return the most likely class
        if log_posterior0 > log_posterior1:
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

    X_PC, Y, pca = prep_mnist(mnist_trainset, 2)
    X_test_PC, Y_test, _ = prep_mnist(mnist_testset, 2, pca)

    # Train the classifier
    mle_clf = MLE_Classifier(X_PC, Y)
    # Predict the test set
    Y_hat = mle_clf.predict(X_test_PC)
    accuracy = compute_accuracy(Y_hat, Y_test)
    error = compute_error(Y_hat, Y_test)
    print("MLE Classifier Error: ", round(error, 7))
    print("MLE Classifier Accuracy: ", round(accuracy, 7))
    print("MLE Classifier mean0 vector: ", mle_clf.mean0)
    print("MLE Classifier mean1 vector: ", mle_clf.mean1)
    print("MLE Classifier covariance0 matrix: ", mle_clf.covariance0)
    print("MLE Classifier covariance1 matrix: ", mle_clf.covariance1)

    # plot the decision boundary
    mle_clf.plot_decision_boundary()

    # print a seperator line
    print("--------------------------------------------------------------------------")

    # Train the classifier
    map_clf = MAP_Classifier(X_PC, Y)
    # Predict the test set
    Y_hat = map_clf.predict(X_test_PC)
    accuracy = compute_accuracy(Y_hat, Y_test)
    error = compute_error(Y_hat, Y_test)
    print("MAP Classifier Error: ", round(error, 7))
    print("MAP Classifier Accuracy: ", round(accuracy, 7))
    print("MAP Classifier mean0 vector: ", map_clf.mean0)
    print("MAP Classifier mean1 vector: ", map_clf.mean1)
    print("MAP Classifier covariance0 matrix: ", map_clf.covariance0)
    print("MAP Classifier covariance1 matrix: ", map_clf.covariance1)
    print("MAP Classifier prior0: ", round(np.exp(map_clf.log_prior0), 7))
    print("MAP Classifier prior1: ", round(np.exp(map_clf.log_prior1), 7))


if __name__ == "__main__":
    main()
