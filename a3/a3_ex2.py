import torchvision.datasets as datasets
import numpy as np

class K_Means_Clustering():
    # take in 784-1 mnist data and run k-means clustering
    def __init__(self, X, K=2):
        self.X = np.array(X)
        self.K = K
        self.centroids = np.random.rand(K, 784)
        self.clusters = np.zeros(len(X))
        self.converged = False
        # train
        self.train()

    def train(self):
        while not self.converged:
            self.clusters = self.vectorized_predict(self.X)
            self.converged = self.vectorized_update_centroids()
        print("Training finished.")

    def vectorized_predict(self, X):
        # vectorized version of predict
        # returns the cluster assignments for each sample in X
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

    def vectorized_update_centroids(self):
        converged = True
        for k in range(self.K):
            X_k = self.X[self.clusters == k]
            if len(X_k) == 0:
                continue
            new_centroid = np.mean(X_k, axis=0)
            if np.linalg.norm(new_centroid - self.centroids[k]) != 0:
                converged = False
            self.centroids[k] = new_centroid
        return converged


    def compute_cluster_consistency(self, Y):
        # compute the consistency of the clusters with the labels
        # return the average consistency.
        # consistency is defined as the number of correctly classified samples in a cluster
        # divided by the total number of samples in the cluster
        total_consistency = 0
        for cluster in range(self.K):
            # find the most common label in the cluster
            labels = [Y[i] for i in range(len(Y)) if self.clusters[i] == cluster]
            if len(labels) == 0:
                continue
            most_common_label = max(set(labels), key=labels.count)
            # compute the consistency
            consistency = len([label for label in labels if label == most_common_label]) / len(labels)
            # print("Cluster {} has consistency {}.".format(cluster, consistency))
            total_consistency += consistency

        return total_consistency / self.K


def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    # prepare the data (flatten the images)
    # use only the first 100 samples per class
    print("Using only the first 100 samples per class. Total number of samples: {}".format(100 * 10))
    filtered_set = []

    for i in range(10):
        total = 0
        for image in mnist_trainset:
            if image[1] == i:
                if total == 100:
                    break
                filtered_set.append(image)
                total += 1

    X = [np.asarray(val[0]).flatten() for val in filtered_set]
    Y = [val[1] for val in filtered_set]

    flattened_trainset = X
    labels_trainset = Y

    # Use total dataset
    # flattened_trainset = [np.reshape(img, (784,)) for img, _ in mnist_trainset]
    # labels_trainset = [label for _, label in mnist_trainset]

    K_vals = [5, 10, 20, 40]

    for K in K_vals:
        k_means = K_Means_Clustering(flattened_trainset, K=K)

        # compute the cluster consistency
        cluster_consistency = k_means.compute_cluster_consistency(labels_trainset)
        print("For K = {} Cluster Consistency: {}".format(K, cluster_consistency))

        print("-" * 50)











if __name__ == "__main__":
    main()




