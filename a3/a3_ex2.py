import sklearn
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from a3_utils import prep_mnist, IMG_PATH, confusion_matrix, compute_accuracy, compute_error
from pathlib import Path


class K_Means_Clustering():
    # take in 784-1 mnist data and run k-means clustering
    def __init__(self, X, K=2):
        self.X = X
        self.K = K
        self.centroids = np.random.rand(K, 784)
        self.clusters = np.zeros(len(X))
        self.converged = False
        # train
        self.train()

    def train(self):
        while not self.converged:
            self.clusters = self.predict(self.X)
            self.converged = self.update_centroids()

    def predict(self, X):
        return np.array([self.classify(x) for x in X])

    def classify(self, x):
        return np.argmin([np.linalg.norm(x - self.centroids[k]) for k in range(self.K)])

    def update_centroids(self):
        converged = True
        for k in range(self.K):
            X_k = [self.X[i] for i in range(len(self.X)) if self.clusters[i] == k]
            if len(X_k) == 0:
                continue
            new_centroid = np.mean(X_k, axis=0)
            if np.linalg.norm(new_centroid - self.centroids[k]) > 1e-5:
                converged = False
            self.centroids[k] = new_centroid
        return converged







def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # prepare the data (flatten the images)
    flattened_trainset = [np.reshape(img, (784,)) for img, _ in mnist_trainset]
    flattened_testset = [np.reshape(img, (784,)) for img, _ in mnist_testset]

    labels_trainset = [label for _, label in mnist_trainset]
    labels_testset = [label for _, label in mnist_testset]

    # run k-means clustering (10 clusters for 10 digits)
    k_means = K_Means_Clustering(flattened_trainset, K=10)

    # assume that the cluster with the most 0s is the cluster for 0s, etc.
    cluster_counts = np.zeros(10)
    for i in range(len(labels_trainset)):
        cluster_counts[k_means.clusters[i]] += 1
    cluster_labels = np.argsort(cluster_counts)[::-1]

    # predict the labels for the test set
    predicted_labels = [cluster_labels[k_means.classify(x)] for x in flattened_testset]

    # compute the accuracy
    accuracy = sklearn.metrics.accuracy_score(labels_testset, predicted_labels)
    print("Accuracy: ", accuracy)











if __name__ == "__main__":
    main()




