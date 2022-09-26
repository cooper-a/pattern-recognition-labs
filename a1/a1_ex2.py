import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from a1_utils import prep_mnist


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

    # def plot_decision_boundary(self):
    #     if self.X_clf is None or self.Y_clf is None:
    #         raise Exception("Classifier not trained")
    #     # plot the decision boundary in 2d
    #     # you can use the following code to plot the prototypes
    #     # plot the line that is equidistant from both prototypes
    #     z1 = np.array(self.prototype0[0], self.prototype0[1])
    #     z2 = np.array(self.prototype1[0], self.prototype1[1])
    #     0 = np.transpose(z1-z2)*(x, y) + 1/2*(np.transpose(z1)*z1 - np.transpose(z2)*z2)
    #     plt.plot(x, y, 'k-')
    #     plt.scatter(self.prototype0[0], self.prototype0[1], marker='x', color='red')
    #     plt.scatter(self.prototype1[0], self.prototype1[1], marker='x', color='blue')
    #     plt.show()

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

    preds_results = []
    for i in range(len(X_test_PC)):
        pred = med_clf.classify(X_test_PC[i])
        preds_results.append((pred, Y_test[i]))

    correct = 0
    for pred in preds_results:
        if pred[0] == pred[1]:
            correct += 1

    print(f"Accuracy for MED Classifier: {round(correct/len(preds_results) * 100, 6)}%")

    # decision boundary for MED PCA 2D
    X_PC_2D, Y = prep_mnist(mnist_trainset, 2)
    med_clf_2D = MED_Classifier(X_PC_2D, Y)
    med_clf_2D.plot_decision_boundary()
    plt.show()
    plt.savefig('med_decision_boundary.png')

    # GED classifier
    ged_clf = GED_Classifier(X_PC, Y)





if __name__ == "__main__":
    main()