import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a1_utils import prep_mnist, IMG_PATH, confusion_matrix
from pathlib import Path

class MED_Classifier:
    def __init__(self, X, Y):
        self.X_clf = X
        self.Y_clf = Y
        self.prototypes = self.__compute_prototypes(X, Y)

    def __compute_prototypes(self, X, Y):
        # Compute the prototypes
        # X is a list of vectors
        # Y is a list of labels
        # returns a list of two vectors, the prototypes
        self.prototype0 = np.mean([X[i] for i in range(len(X)) if Y[i] == 0], axis=0)
        self.prototype1 = np.mean([X[i] for i in range(len(X)) if Y[i] == 1], axis=0)
        return [self.prototype0, self.prototype1]

    def __check_if_clf_trained(self):
        if self.X_clf is None or self.Y_clf is None:
            raise Exception("Classifier not trained")

    def classify(self, x):
        # Implement the MED classifier
        # x is a single vector to classify
        # returns the classification of x by nearest euclidean distance to prototype
        self.__check_if_clf_trained()
        dist0 = np.linalg.norm(self.prototype0 - x)
        dist1 = np.linalg.norm(self.prototype1 - x)
        if dist0 < dist1:
            return 0
        else:
            return 1

    def plot_decision_boundary(self, h=5):
        self.__check_if_clf_trained()
        if self.X_clf.shape[1] != 2:
            raise Exception("Decision boundary can only be plotted for 2D data")
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
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 0')
        plt.legend(handles=[red_patch, blue_patch])
        plt.title('MED decision boundary')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        path = IMG_PATH + 'MED_decision_boundary.png'
        plt.savefig(path)
        plt.show()

    def plot_decision_boundary_analytical(self):
        self.__check_if_clf_trained()
        if self.X_clf.shape[1] != 2:
            raise Exception("Decision boundary can only be plotted for 2D data")
        weights, bias = self.determine_decision_boundary_analytical()
        w1 = weights[0][0]
        w2 = weights[0][1]
        bias = bias[0]
        x = np.linspace(self.X_clf[:, 0].min(), self.X_clf[:, 0].max(), 100)
        y = (-1 * bias / w2) - (w1 / w2) * x
        plt.plot(x, y, 'r')
        plt.scatter(self.X_clf[:, 0], self.X_clf[:, 1], c=self.Y_clf, cmap=plt.cm.coolwarm)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.xlim(self.X_clf[:, 0].min(), self.X_clf[:, 0].max())
        plt.ylim(self.X_clf[:, 1].min(), self.X_clf[:, 1].max())
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 0')
        plt.legend(handles=[red_patch, blue_patch])
        plt.title('MED decision boundary analytical')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        path = IMG_PATH + 'MED_decision_boundary_analytical.png'
        plt.savefig(path)
        plt.show()

    def determine_decision_boundary_analytical(self):
        # in form of w1x1 + w2x2 + b = 0
        self.__check_if_clf_trained()
        prototype0 = self.prototype0.reshape(self.prototype0.shape[0], 1)
        prototype1 = self.prototype1.reshape(self.prototype0.shape[0], 1)
        weights = (prototype0 - prototype1).T
        bias = 0.5 * (np.matmul(prototype1.T, prototype1) - np.matmul(prototype0.T, prototype0))
        return weights, bias

    def print_decision_boundary_analytical(self):
        self.__check_if_clf_trained()
        weights, bias = self.determine_decision_boundary_analytical()
        equation_string = "Decision boundary equation: "
        for i in range(len(weights[0])):
            equation_string += f"{round(weights[0][i], 3)}x{i + 1} + "
        equation_string += f"{round(bias[0][0], 3)} = 0"
        print(equation_string)




class GED_Classifier:
    def __init__(self, X, Y):
        # compute covariance matrix
        c = np.cov(X, rowvar=False)
        # compute the mean of each class
        self.mean0 = np.mean([X[i] for i in range(len(X)) if Y[i] == 0], axis=0)
        self.mean1 = np.mean([X[i] for i in range(len(X)) if Y[i] == 1], axis=0)
        # compute the eigenvalues and eigenvectors of the covariance matrix
        self.eigvals, self.eigvecs = np.linalg.eig(c)
        # compute the whitening matrix
        self.whitening_matrix = np.dot(np.diag(1 / np.sqrt(self.eigvals)), self.eigvecs.T)
        # compute the mean of each class in the whitened space
        self.mean0_w = np.dot(self.whitening_matrix, self.mean0)
        self.mean1_w = np.dot(self.whitening_matrix, self.mean1)
        # compute the covariance matrix of each class in the whitened space



def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    X_PC, Y = prep_mnist(mnist_trainset, 20)
    X_test_PC, Y_test = prep_mnist(mnist_testset, 20)

    # MED classifier
    med_clf = MED_Classifier(X_PC, Y)

    med_clf.print_decision_boundary_analytical()
    preds_results = []
    for i in range(len(X_test_PC)):
        pred = med_clf.classify(X_test_PC[i])
        preds_results.append((pred, Y_test[i]))

    correct = 0
    for pred in preds_results:
        if pred[0] == pred[1]:
            correct += 1
    print(f"Accuracy for 20D MED Classifier: {round(correct/len(preds_results) * 100, 3)}%")


    # decision boundary for MED PCA 2D
    X_PC_2D, Y = prep_mnist(mnist_trainset, 2)
    X_test_PC_2D, Y_test = prep_mnist(mnist_testset, 2)
    med_clf_2D = MED_Classifier(X_PC_2D, Y)
    print("Plotting decision boundary for 2D MED Classifier")
    med_clf_2D.plot_decision_boundary_analytical()
    med_clf_2D.plot_decision_boundary()
    med_clf_2D.print_decision_boundary_analytical()

    preds_results = []
    for i in range(len(X_test_PC_2D)):
        pred = med_clf_2D.classify(X_test_PC_2D[i])
        preds_results.append((pred, Y_test[i]))

    correct = 0
    for pred in preds_results:
        if pred[0] == pred[1]:
            correct += 1

    print(f"Accuracy for 2D MED Classifier: {round(correct / len(preds_results) * 100, 3)}%")

    # # GED classifier
    # ged_clf = GED_Classifier(X_PC, Y)




if __name__ == "__main__":
    main()