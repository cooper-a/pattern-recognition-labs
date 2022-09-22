import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA


class MED_Classifier:
    def __init__(self, X, Y):
        self.X_clf = X
        self.Y_clf = Y
        self.prototype0 = np.mean(X[Y == 0], axis=0)
        self.prototype1 = np.mean(X[Y == 1], axis=0)

    def classify(self, x):
        # Implement the MED classifier
        # x is a single vector to classify
        # returns the classification of x by nearest euclidean distance to prototype
        if self.X_clf is None or self.Y_clf is None:
            raise Exception("Classifier not trained")
        dist0 = np.linalg.norm(self.prototype0 - x)
        dist1 = np.linalg.norm(self.prototype1 - x)
        if dist0 < dist1:
            return 0
        else:
            return 1