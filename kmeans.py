"""
Kmeans Algorithm Module
"""

# import necessary modules
import numpy as np


class Kmeans():
    """
    Kmeans Algorithm Class
    """

    # initialize kmeans
    def __init__(self, n_clusters, max_epochs, intra_cluster_variance):
        self.n_clusters = n_clusters  # number of clusters
        self.max_epochs = max_epochs  # max number of epochs for training
        self.centroids = np.zeros(n_clusters)  # centroid of each cluster
        self.variance = np.zeros(n_clusters)  # variance of each cluster
        self.intra_cluster_variance = intra_cluster_variance
        self.sse = 0  # sum of squared errors

    def initialize_centroids(self, features):
        """
        initaliaze the centroids of each cluster
        """

        centroids = np.random.permutation(features)[0:self.n_clusters]
        print(centroids)
        return centroids

    def compute_distance(self, features, centroids):
        """
        compute the distances between the features and the centroids
        """

        distance = np.zeros((features.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(features - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_centroids(self, features, targets):
        """
        compute the centroids
        """

        centroids = np.zeros((self.n_clusters, features.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(features[targets == k, :], axis=0)
        return centroids

    def find_closest_cluster(self, distance):
        """
        find the closest clusters
        """

        return np.argmin(distance, axis=1)

    def compute_sse(self, features, clusters, centroids):
        """
        compute the sum of the squared errors
        """

        distance = np.zeros(features.shape[0])
        for k in range(self.n_clusters):
            distance[clusters == k] = np.linalg.norm(
                features[clusters == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def compute_variance(self, features, clusters):
        """
        compute the variance of each cluster
        """

        variance = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            variance[k] = np.var(features[clusters == k])

        # if variance = 0, reassign value to mean of the variances
        variance[variance == 0] = np.mean(variance)
        return variance

    def compute_constant_variance(self, centroids):
        """
        compute the constant variance
        """

        max_distance = 0
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                distance = np.linalg.norm(centroids[i] - centroids[j])

                if distance > max_distance:
                    max_distance = distance

        variance = np.ones(self.n_clusters) * \
            np.square(max_distance / np.sqrt(2 * self.n_clusters))

        return variance

    def train(self, features):
        """
        train the kmeans algorithm
        """

        self.centroids = self.initialize_centroids(features)
        for i in range(self.max_epochs):
            old_centroids = self.centroids
            distance = self.compute_distance(features, old_centroids)
            clusters = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(features, clusters)

            if np.all(old_centroids == self.centroids):
                break

        # self.sse = self.compute_sse(features, clusters, self.centroids)

        if self.intra_cluster_variance is True:
            # compute intra cluster variance
            self.variance = self.compute_variance(features, clusters)
        else:
            # compute constant variance
            self.variance = self.compute_constant_variance(self.centroids)

    def determine_labels(self, features, targets):
        distance = self.compute_distance(features, self.centroids)
        clusters = self.find_closest_cluster(distance)
        print(clusters.shape)
        print(clusters)

        thingy = np.zeros((self.n_clusters, 1))
        for index in clusters[clusters == 0]:
            thingy[targets[index]] += 1

        print(len(clusters[clusters == 0]))
        print(clusters[clusters == 0])
        print(thingy)
        print(np.max(thingy))
        print(np.argmax(thingy))
