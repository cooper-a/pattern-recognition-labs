import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from a3_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class Histogram_Classifier():
    # Implement a non-parametric classifier using histograms
    def __init__(self, X, Y, bin_width=1):
        self.X_clf = X
        self.Y_clf = Y
        self.bin_width = bin_width
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        self.X0 = X0
        self.X1 = X1
        self.h_prob0, self.h_min_0, self.h_max_0 = self.fit(np.array(X0), bin_width)
        self.h_prob1, self.h_min_1, self.h_max_1 = self.fit(np.array(X1), bin_width)

    def fit(self, X, bin_width):
        # function definition taken from tutorial 5
        minX = X.min()
        maxX = X.max()
        number_bins = int(np.ceil((maxX - minX) / bin_width))
        h_min = minX
        h_max = minX + number_bins * bin_width

        j_indices = np.floor((X - minX) / bin_width).astype(int)
        M = np.zeros(number_bins, dtype=np.float32)
        for j in j_indices:
            M[j] += 1
        h_probs = M / (len(X) * bin_width)
        return h_probs, h_min, h_max

    def predict_histogram(self, X_test, h_prob, h_min, h_max, bin_width):
        # function definition taken from tutorial 5
        non_zero_idx = (X_test >= h_min) & (X_test < h_max)
        j_indices = np.floor((X_test[non_zero_idx] - h_min) / bin_width).astype(int)
        p_hat = np.zeros(len(X_test), dtype=np.float32)
        p_hat[non_zero_idx] = h_prob[j_indices]
        return p_hat

    def predict(self, X):
        X = np.array(X)
        X = X.reshape(-1,)
        prob0 = self.predict_histogram(X, self.h_prob0, self.h_min_0, self.h_max_0, self.bin_width)
        prob1 = self.predict_histogram(X, self.h_prob1, self.h_min_1, self.h_max_1, self.bin_width)
        return np.array([0 if prob0[i] >= prob1[i] else 1 for i in range(len(X))])

    def __check_if_clf_trained(self):
        if self.h_prob0 is None or self.h_prob1 is None:
            raise Exception("Classifier is not trained yet.")

    def plot_histogram(self):
        # with probability as y-axis
        self.__check_if_clf_trained()
        total_min = min(min(self.X0), min(self.X1))[0] - 2 * self.bin_width
        total_max = max(max(self.X0), max(self.X1))[0] + 2 * self.bin_width
        x_plot = np.linspace(total_min, total_max, 5000)

        preds_0 = self.predict_histogram(x_plot, self.h_prob0, self.h_min_0, self.h_max_0, self.bin_width)
        preds_1 = self.predict_histogram(x_plot, self.h_prob1, self.h_min_1, self.h_max_1, self.bin_width)
        plt.plot(x_plot, preds_0, '-r', label=r'$\hatp(x|0)$, histogram')
        plt.plot(x_plot, preds_1, '-b', label=r'$\hatp(x|1)$, histogram')
        plt.xlabel("$x$")
        plt.ylabel(r"$\hatp$")
        plt.title("Histogram of class 0 and class 1 in 1D Space bin_width={}".format(self.bin_width))
        plt.legend(loc='upper right')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(IMG_PATH + "histogram_bin_width_{}.png".format(self.bin_width))
        plt.show()


class KDE_Classifier():
    # Implement a non-parametric classifier using kernel density estimation with a Gaussian kernel
    def __init__(self, X, Y, sigma=20):
        self.X_clf = X
        self.Y_clf = Y
        X0 = [X[i] for i in range(len(X)) if Y[i] == 0]
        X1 = [X[i] for i in range(len(X)) if Y[i] == 1]
        self.X0 = np.array(X0)
        self.X1 = np.array(X1)
        self.sigma = sigma

    def gaussian_window(self, x):
        # unscaled gaussian window
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def prob0(self, x):
        self.__check_if_clf_trained()
        vals = (self.X0 - x) / self.sigma
        return np.mean(self.gaussian_window(vals) / self.sigma)
    def prob1(self, x):
        self.__check_if_clf_trained()
        vals = (self.X1 - x) / self.sigma
        return np.mean(self.gaussian_window(vals) / self.sigma)

    def classify(self, x):
        return 0 if self.prob0(x) > self.prob1(x) else 1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.X0 is None or self.X1 is None:
            raise Exception("Classifier is not trained yet.")

    def plot_parzen_distribution(self):
        # plot the PDF of the two classes with probability as y-axis
        self.__check_if_clf_trained()
        total_min = min(min(self.X0), min(self.X1))[0] - 2 * self.sigma
        total_max = max(max(self.X0), max(self.X1))[0] + 2 * self.sigma
        x = np.arange(total_min, total_max, 0.1)
        y0 = np.zeros(len(x))
        y1 = np.zeros(len(x))
        for i in range(len(x)):
            y0[i] = self.prob0(np.array([x[i]]))
            y1[i] = self.prob1(np.array([x[i]]))

        plt.figure()
        plt.plot(x, y0, color='r', label=r'$\hatp(x|0)$, KDE')
        plt.plot(x, y1, color='b', label=r'$\hatp(x|1)$, KDE')
        plt.xlabel("$x$")
        plt.ylabel(r"$\hatp$")
        plt.title("Parzen Distribution of class 0 and class 1 in 1D Space sigma={}".format(self.sigma))
        plt.legend(loc='upper right')
        Path(IMG_PATH).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(IMG_PATH + "parzen_distribution.png")
        plt.show()




def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    X_PC, Y, pca = prep_mnist(mnist_trainset, n_components=1)
    X_test_PC, Y_test, _ = prep_mnist(mnist_testset, n_components=1, pca=pca)

    clf = Histogram_Classifier(X_PC, Y, bin_width=1)
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




