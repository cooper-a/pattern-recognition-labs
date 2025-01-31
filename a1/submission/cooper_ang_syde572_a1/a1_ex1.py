import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a1_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class kNNClassifier:
    def __init__(self, X, Y):
        self.X_clf = np.array(X)
        self.Y_clf = np.array(Y)

    def classify(self, x, k, dist_func=np.linalg.norm, decision_weighted=True):
        # Implement the k-NN classifier
        # x is a single vector to classify
        # k is the number of neighbors to consider
        # returns the most common class among the k nearest neighbors
        self.__check_if_clf_trained()
        # Vectorized implementation for speed
        distances = dist_func(self.X_clf - x, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = self.Y_clf[k_nearest_indices]
        # count the number of 0s and 1s
        count0 = 0
        count1 = 0
        # protect against divide by zero
        epsilon = 1e-10
        if decision_weighted:
            for i in range(len(k_nearest_indices)):
                if k_nearest_labels[i] == 0:
                    count0 += 1 / (distances[k_nearest_indices[i]] + epsilon)
                else:
                    count1 += 1 / (distances[k_nearest_indices[i]] + epsilon)
        else:
            for i in range(len(k_nearest_indices)):
                if k_nearest_labels[i] == 0:
                    count0 += 1
                else:
                    count1 += 1
        # return the most common class
        if count0 > count1:
            return 0
        else:
            return 1

    def __check_if_clf_trained(self):
        if self.X_clf is None or self.Y_clf is None:
            raise Exception("Classifier not trained")

    def predict(self, X, k, dist_func=np.linalg.norm, decision_weighted=True):
        self.__check_if_clf_trained()
        return [self.classify(x, k, dist_func=dist_func, decision_weighted=decision_weighted) for x in X]

    def plot_decision_boundary(self, k, dist_func=np.linalg.norm, decision_weighted=True, h=25):
        self.__check_if_clf_trained()
        x_min, x_max = self.X_clf[:, 0].min() - 1, self.X_clf[:, 0].max() + 1
        y_min, y_max = self.X_clf[:, 1].min() - 1, self.X_clf[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                Z[i, j] = self.classify(np.array([xx[i, j], yy[i, j]]), k, dist_func, decision_weighted)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(self.X_clf[:, 0], self.X_clf[:, 1], c=self.Y_clf, cmap=plt.cm.coolwarm)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"{k}-NN decision boundary")

        # legend
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 0')
        plt.legend(handles=[red_patch, blue_patch])
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        path = IMG_PATH + f"{k}-NN_decision_boundary.png"
        plt.savefig(path)
        plt.show()


K_VALUES = (1, 2, 3, 4, 5)


def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    X_PC, Y = prep_mnist(mnist_trainset, 2)
    X_test_PC, Y_test = prep_mnist(mnist_testset, 2)

    knn = kNNClassifier(X_PC, Y)
    errors = {}
    for k_val in K_VALUES:
        knn.plot_decision_boundary(k_val)
        Y_hat = knn.predict(X_test_PC, k_val)
        accuracy = compute_accuracy(Y_test, Y_hat)
        error = compute_error(Y_test, Y_hat)
        cf = confusion_matrix(Y_test, Y_hat)
        print(f"kNN with k={k_val} is {round(accuracy * 100, 3)}% accurate")
        print(f"kNN with k={k_val} has an error of {round(error, 6)}")
        print(f"Confusion matrix for kNN with k={k_val}: \n{cf}")
        errors[k_val] = error

    # plot a bar graph of the errors
    plt.bar(errors.keys(), errors.values())
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title("Error vs. k for kNN classifier")
    path = IMG_PATH + f"knn_error.png"
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()
