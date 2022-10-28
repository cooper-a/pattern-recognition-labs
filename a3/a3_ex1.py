import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a3_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class Histogram_Classifier():
    # Implement a non-parametric classifier using histograms
    def __init__(self, X, Y, bin_width=0.1):
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
        if x[0] < self.min0 or x[0] > self.max0:
            prob0 = 0
        else:
            prob0 = self.proba0[int(np.floor((x[0] - self.min0) / self.bin_width))]
        if x[0] < self.min1 or x[0] > self.max1:
            prob1 = 0
        else:
            prob1 = self.proba1[int(np.floor((x[0] - self.min1) / self.bin_width))]
        if prob0 > prob1:
            return 0
        else:
            return 1

    def predict(self, X):
        return [self.classify(x) for x in X]

    def __check_if_clf_trained(self):
        if self.hist0 is None or self.hist1 is None:
            raise Exception("Classifier is not trained yet.")

    def plot_histogram(self):
        


