import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a1_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
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

    def predict(self, X):
        # X is a list of data points
        self.__check_if_clf_trained()
        return [self.classify(x) for x in X]

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
        plt.contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.8)
        plt.scatter(self.X_clf[:, 0], self.X_clf[:, 1], c=self.Y_clf, cmap=plt.cm.bwr)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 0')
        plt.legend(handles=[red_patch, blue_patch])
        plt.title('MED Decision Boundary (2D)')
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
        boundary = plt.plot(x, y, 'k', label='Decision boundary')
        plt.scatter(self.X_clf[:, 0], self.X_clf[:, 1], c=self.Y_clf, cmap=plt.cm.bwr)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.xlim(self.X_clf[:, 0].min(), self.X_clf[:, 0].max())
        plt.ylim(self.X_clf[:, 1].min(), self.X_clf[:, 1].max())
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 0')
        plt.legend(handles=[red_patch, blue_patch, boundary[0]])
        plt.title('MED Decision Boundary Analytical (2D)')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        path = IMG_PATH + 'MED_decision_boundary_analytical.png'
        plt.savefig(path)
        plt.show()

    def determine_decision_boundary_analytical(self):
        # in form of w1x1 + w2x2 + b = 0
        self.__check_if_clf_trained()
        prototype0 = self.prototype0.reshape(self.prototype0.shape[0], 1)
        prototype1 = self.prototype1.reshape(self.prototype0.shape[0], 1)
        weights = (prototype1 - prototype0).T
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
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        # compute the mean of each class
        self.mean0 = np.mean(X0, axis=0)
        self.mean1 = np.mean(X1, axis=0)
        covariance0 = np.cov(X0, rowvar=False)
        covariance1 = np.cov(X1, rowvar=False)

        # compute the inverse covariance with eigenvalue and eigenvector decomposition
        eigvals0, eigvecs0 = np.linalg.eig(covariance0)
        weight_matrix0 = np.diag(eigvals0 ** (-1/2)) @ eigvecs0.T
        self.covariance0_inverse = weight_matrix0.T @ weight_matrix0

        eigvals1, eigvecs1 = np.linalg.eig(covariance1)
        weight_matrix1 = np.diag(eigvals1 ** (-1/2)) @ eigvecs1.T
        self.covariance1_inverse = weight_matrix1.T @ weight_matrix1

        # covariance0_inverse = np.linalg.inv(covariance0)
        # covariance1_inverse = np.linalg.inv(covariance1)
        # self.covariance0_inverse = covariance0_inverse
        # self.covariance1_inverse = covariance1_inverse
        # # sanity check if the inverse covariance is the same
        # assert np.allclose(self.covariance0_inverse, covariance0_inverse)
        # assert np.allclose(self.covariance1_inverse, covariance1_inverse)

        self.X_clf = X
        self.Y_clf = Y

    def __check_if_clf_trained(self):
        if self.X_clf is None or self.Y_clf is None:
            raise Exception("Classifier not trained")

    def classify(self, x):
        dist0 = np.sqrt(((x - self.mean0).T @ self.covariance0_inverse @ (x - self.mean0)))
        dist1 = np.sqrt(((x - self.mean1).T @ self.covariance1_inverse @ (x - self.mean1)))
        if dist0 < dist1:
            return 0
        else:
            return 1

    def predict(self, X):
        # X is a list of data points
        self.__check_if_clf_trained()
        return [self.classify(x) for x in X]

    def decision_boundary_fct(self, x1, x2):
        x = np.array([x1, x2])
        return np.sqrt(((x - self.mean0).T @ self.covariance0_inverse @ (x - self.mean0))) - np.sqrt(
            ((x - self.mean1).T @ self.covariance1_inverse @ (x - self.mean1)))

    def plot_decision_boundary_analytical(self, h=5):
        if self.X_clf.shape[1] != 2:
            raise Exception("Decision boundary can only be plotted for 2D data")
        x_min, x_max = self.X_clf[:, 0].min() - 1, self.X_clf[:, 0].max() + 1
        y_min, y_max = self.X_clf[:, 1].min() - 1, self.X_clf[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                Z[i, j] = self.decision_boundary_fct(xx[i, j], yy[i, j])
        decision_boundary = plt.contour(xx, yy, Z, levels=(0,), colors='k')
        boundary, _ = decision_boundary.legend_elements()
        plt.scatter(self.X_clf[:, 0], self.X_clf[:, 1], c=self.Y_clf, cmap=plt.cm.bwr)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        red_patch = mpatches.Patch(color='red', label='Class 0')
        blue_patch = mpatches.Patch(color='blue', label='Class 1')
        plt.legend(handles=[red_patch, blue_patch, boundary[0]], labels=['Class 1', 'Class 0', 'Decision boundary'])
        plt.title('GED Decision Boundary Analytical (2D)')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        path = IMG_PATH + 'GED_decision_boundary_analytical.png'
        plt.savefig(path)
        plt.show()

    def determine_decision_boundary_analytical(self):
        self.__check_if_clf_trained()
        Q0 = self.covariance0_inverse - self.covariance1_inverse
        Q1 = 2 * (self.mean1.T @ self.covariance1_inverse - self.mean0.T @ self.covariance0_inverse)
        Q2 = self.mean0.T @ self.covariance0_inverse @ self.mean0 - self.mean1.T @ self.covariance1_inverse @ self.mean1
        return Q0, Q1, Q2

    def print_decision_boundary_analytical(self):
        self.__check_if_clf_trained()
        Q0, Q1, Q2 = self.determine_decision_boundary_analytical()
        equation_string = "Decision boundary equation: "
        for i in range(Q0.shape[0]):
            for j in range(Q0.shape[1]):
                if Q0[i, j] != 0:
                    equation_string += f"{round(Q0[i, j], 6)}x{i + 1}x{j + 1} + "
        for i in range(Q1.shape[0]):
            if Q1[i] != 0:
                equation_string += f"{round(Q1[i], 6)}x{i + 1} + "
        equation_string += f"{round(Q2, 6)} = 0"
        print(equation_string)

def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    X_PC, Y = prep_mnist(mnist_trainset, 20)
    X_test_PC, Y_test = prep_mnist(mnist_testset, 20)

    # MED classifier 20D
    med_clf = MED_Classifier(X_PC, Y)
    w, b = med_clf.determine_decision_boundary_analytical()
    med_clf.print_decision_boundary_analytical()
    Y_hat = med_clf.predict(X_test_PC)
    accuracy = compute_accuracy(Y_hat, Y_test)
    cf = confusion_matrix(Y_hat, Y_test)
    med_error = compute_error(Y_hat, Y_test)
    print(f"Accuracy for 20D MED Classifier: {round(accuracy * 100, 3)}%")
    print(f"Error for 20D MED Classifier: {round(med_error, 6)}")
    print(f"Confusion matrix for 20D MED Classifier: \n{cf}")

    # MED classifier 2D
    X_PC_2D, Y = prep_mnist(mnist_trainset, 2)
    X_test_PC_2D, Y_test = prep_mnist(mnist_testset, 2)
    med_clf_2D = MED_Classifier(X_PC_2D, Y)
    # print("Plotting decision boundary for 2D MED Classifier")
    med_clf_2D.plot_decision_boundary_analytical()
    med_clf_2D.plot_decision_boundary()
    med_clf_2D.print_decision_boundary_analytical()

    Y_hat = med_clf_2D.predict(X_test_PC_2D)
    accuracy = compute_accuracy(Y_hat, Y_test)
    cf = confusion_matrix(Y_test, Y_hat)
    error = compute_error(Y_hat, Y_test)
    print(f"Accuracy for 2D MED Classifier: {round(accuracy * 100, 3)}%")
    print(f"Error for 2D MED Classifier: {round(error, 6)}")
    print(f"Confusion matrix for 2D MED Classifier: \n{cf}")

    # GED classifier 20D
    ged_clf = GED_Classifier(X_PC, Y)

    Q0, Q1, Q2 = ged_clf.determine_decision_boundary_analytical()

    Y_hat = ged_clf.predict(X_test_PC)
    accuracy = compute_accuracy(Y_hat, Y_test)
    cf = confusion_matrix(Y_test, Y_hat)
    ged_error = compute_error(Y_hat, Y_test)
    print(f"Accuracy for 20D GED Classifier: {round(accuracy * 100, 3)}%")
    print(f"Error for 20D GED Classifier: {round(ged_error, 6)}")
    print(f"Confusion matrix for 20D GED Classifier: \n{cf}")
    plt.bar(["MED", "GED"], [med_error, ged_error])
    plt.title("MED vs GED Classifier Error")
    plt.xlabel("Classifier")
    plt.ylabel("Error")
    path = IMG_PATH + f"MED_vs_GED_Error.png"
    plt.savefig(path)
    plt.show()
    plt.clf()

    # GED classifier 2D
    ged_clf_2D = GED_Classifier(X_PC_2D, Y)
    # print("Plotting decision boundary for 2D GED Classifier")
    ged_clf_2D.plot_decision_boundary_analytical()
    ged_clf_2D.print_decision_boundary_analytical()
    Y_hat = ged_clf_2D.predict(X_test_PC_2D)
    accuracy = compute_accuracy(Y_hat, Y_test)
    cf = confusion_matrix(Y_test, Y_hat)
    error = compute_error(Y_hat, Y_test)
    print(f"Accuracy for 2D GED Classifier: {round(accuracy * 100, 3)}%")
    print(f"Error for 2D GED Classifier: {round(error, 6)}")
    print(f"Confusion matrix for 2D GED Classifier: \n{cf}")


if __name__ == "__main__":
    main()
