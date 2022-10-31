import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a3_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class Histogram_Classifier():
    # Implement a non-parametric classifier using histograms
    def __init__(self, X, Y, bin_width=1):
        self.X_clf = X
        self.Y_clf = Y
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        self.X0 = X0
        self.X1 = X1

        # manually create histogram without using np.histogram
        self.bin_width = bin_width
        self.max0, self.min0, self.max1, self.min1 = max(X0)[0], min(X0)[0], max(X1)[0], min(X1)[0]
        number_of_bins0 = int(np.ceil((self.max0 - self.min0) / bin_width))
        number_of_bins1 = int(np.ceil((self.max1 - self.min1) / bin_width))
        self.hist0 = np.zeros(number_of_bins0)
        self.hist1 = np.zeros(number_of_bins1)
        for x in X0:
            self.hist0[int(np.floor((x[0] - self.min0) / bin_width))] += 1
        for x in X1:
            self.hist1[int(np.floor((x[0] - self.min1) / bin_width))] += 1

        # convert to probability
        self.proba0 = self.hist0 / (len(X0) * bin_width)
        self.proba1 = self.hist1 / (len(X1) * bin_width)


    def classify(self, x):
        return 0 if self.prob0(x) > self.prob1(x) else 1

    def prob0(self, x):
        if x[0] < self.min0 or x[0] > self.max0:
            return 0
        else:
            return self.proba0[int(np.floor((x[0] - self.min0) / self.bin_width))]

    def prob1(self, x):
        if x[0] < self.min1 or x[0] > self.max1:
            return 0
        else:
            return self.proba1[int(np.floor((x[0] - self.min1) / self.bin_width))]

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.hist0 is None or self.hist1 is None:
            raise Exception("Classifier is not trained yet.")

    def plot_histogram(self):
        # with probability as y-axis
        self.__check_if_clf_trained()
        plt.figure()
        plt.bar(np.arange(self.min0, self.max0, self.bin_width), self.proba0, width=self.bin_width, color='b')
        plt.bar(np.arange(self.min1, self.max1, self.bin_width), self.proba1, width=self.bin_width, color='r')
        plt.xlabel("PC1")
        plt.ylabel("Probability")
        plt.title("Histogram of class 0 and class 1 in 1D Space bin_width={}".format(self.bin_width))
        plt.legend(handles=[mpatches.Patch(color='b', label='Class 0'), mpatches.Patch(color='r', label='Class 1')])
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        plt.savefig(IMG_PATH + "histogram.png")
        plt.show()


class KDE_Classifier():
    # Implement a non-parametric classifier using kernel density estimation with a Gaussian kernel
    def __init__(self, X, Y, sigma=20):
        self.X_clf = X
        self.Y_clf = Y
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        self.X0 = X0
        self.X1 = X1
        self.sigma = sigma

    def classify(self, x):
        # vectorized implementation
        prob0 = self.parzen_class0(x)
        prob1 = self.parzen_class1(x)
        if prob0 > prob1:
            return 0
        else:
            return 1

    def parzen_class0(self, x):
        prob0 = np.sum(np.exp(-np.sum((self.X0 - x) ** 2, axis=1) / (2 * self.sigma ** 2))) / (len(self.X0) * 1/np.sqrt(2*np.pi*self.sigma**2))
        return prob0

    def parzen_class1(self, x):
        prob1 = np.sum(np.exp(-np.sum((self.X1 - x) ** 2, axis=1) / (2 * self.sigma ** 2))) / (len(self.X1) * 1/np.sqrt(2*np.pi*self.sigma**2))
        return prob1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.X0 is None or self.X1 is None:
            raise Exception("Classifier is not trained yet.")

    def plot_parzen_distribution(self):
        # plot the PDF of the two classes with probability as y-axis
        self.__check_if_clf_trained()
        total_min = min(min(self.X0), min(self.X1))[0]
        total_max = max(max(self.X0), max(self.X1))[0]
        x = np.arange(total_min, total_max, 0.1)
        y0 = np.zeros(len(x))
        y1 = np.zeros(len(x))
        for i in range(len(x)):
            y0[i] = self.parzen_class0([x[i]])
            y1[i] = self.parzen_class1([x[i]])
        plt.figure()
        plt.plot(x, y0, color='blue')
        plt.plot(x, y1, color='orange')
        plt.xlabel("PC1")
        plt.ylabel("Probability")
        plt.title("Parzen Distribution of class 0 and class 1 in 1D Space sigma={}".format(self.sigma))
        plt.legend(handles=[mpatches.Patch(color='blue', label='0'), mpatches.Patch(color='orange', label='1')])
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        plt.savefig(IMG_PATH + "parzen_distribution.png")
        plt.show()




def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    X_PC, Y, pca = prep_mnist(mnist_trainset, n_components=1)
    X_test_PC, Y_test, _ = prep_mnist(mnist_testset, n_components=1, pca=pca)

    clf = Histogram_Classifier(X_PC, Y)
    clf.plot_histogram()
    Y_pred = clf.predict(X_test_PC)
    print("Histogram Classifier with bin width = 1")
    print("Accuracy: ", compute_accuracy(Y_test, Y_pred))
    print("Error: ", compute_error(Y_test, Y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, Y_pred))

    print("-" * 50)

    clf = Histogram_Classifier(X_PC, Y, bin_width=10)
    clf.plot_histogram()
    Y_pred = clf.predict(X_test_PC)
    print("Histogram Classifier with bin width = 10")
    print("Accuracy: ", compute_accuracy(Y_test, Y_pred))
    print("Error: ", compute_error(Y_test, Y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, Y_pred))

    print("-" * 50)

    clf = Histogram_Classifier(X_PC, Y, bin_width=100)
    clf.plot_histogram()
    Y_pred = clf.predict(X_test_PC)
    print("Histogram Classifier with bin width = 100")
    print("Accuracy: ", compute_accuracy(Y_test, Y_pred))
    print("Error: ", compute_error(Y_test, Y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, Y_pred))

    print("-" * 50)

    clf = KDE_Classifier(X_PC, Y, sigma=20)
    clf.plot_parzen_distribution()
    Y_pred = clf.predict(X_test_PC)
    print("KDE Classifier with sigma = 20")
    print("Accuracy: ", compute_accuracy(Y_test, Y_pred))
    print("Error: ", compute_error(Y_test, Y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, Y_pred))







if __name__ == "__main__":
    main()




