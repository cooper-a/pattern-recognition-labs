import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a2_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class MLE_Classifier:
    # Implement the MLE classifier with exponential
    def __init__(self, X, Y):

    def classify(self, x):


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
    print(f"Accuracy: {accuracy}")
    error = compute_error(Y_hat, Y_test)
    print(f"Error: {error}")

    # plot the decision boundary
    mle_clf.plot_decision_boundary()

    # Train the classifier
    map_clf = MAP_Classifier(X_PC, Y)
    # Predict the test set
    Y_hat = map_clf.predict(X_test_PC)
    accuracy = compute_accuracy(Y_hat, Y_test)
    print(f"Accuracy: {accuracy}")
    error = compute_error(Y_hat, Y_test)
    print(f"Error: {error}")


if __name__ == "__main__":
    main()
