"""
K-Means Classifier Module
"""

# import necessary modules
import numpy as np


class KMeansClassifier():
    """
    K-Means Classifier Class

    This class initializes and trains a K-Means classifier for a given number
    of clusters. This class will find an arbitrary number of clusters that
    maximizes intracluster similarity and minimizes intercluster similarity
    according to the provided feature data.
    """

    def __init__(self, n_clusters=1):

        # set the given number of clusters
        self.n_clusters = n_clusters

    def initialize_centroids(self, features):
        """
        Initialize Centroids

        This method will randomly initialize the initial centroids of the
        K-Means clustering algorithm to random data points contained within the
        features matrix

        Args:
            features (np.array): the features matrix

        Returns:
            centroids (np.array): the initial centroids
        """

        centroids = np.random.permutation(features)[0:self.n_clusters]

        return centroids

    def compute_distance(self, features, centroids):
        """
        Compute Distance

        This method will compute the distances of all the features to the
        centroids using Euclidean distance.

        Args:
            features (np.array): the features matrix
            centroids (np.array): the centroids of each cluster

        Returns:
            distance (np.array): the distances from each feature to each cluster
        """

        distance = np.zeros((features.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(features - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)

        return distance

    def compute_centroids(self, features, clusters):
        """
        Compute Centroids

        This method will compute the centroids of each cluster.

        Args:
            features (np.array): the features matrix
            clusters (np.array): the assigned clusters of each feature

        Returns:
            centroids (np.array): the centroids
        """

        centroids = np.zeros((self.n_clusters, features.shape[1]))

        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(features[clusters == k, :], axis=0)

        return centroids

    def find_closest_cluster(self, distance):
        """
        Find Closest Cluster

        This method will find the closest cluster of each feature provided by
        the distance to each cluster from each feature.

        Args:
            distance (np.array): the distance matrix

        Returns:
            clusters (np.array): the closest clusters
        """

        clusters = np.argmin(distance, axis=1)

        return clusters

    def train(self, features, n_epochs=1):
        """
        Train

        This method will train the K-Means Classifier by iterating over the
        provided features for a given number of epochs or until convergence.

        Args:
            features (np.array): the features matrix
            n_epochs (int): the number of epochs to train
        """

        # initialize centroids
        self.centroids = self.initialize_centroids(features)

        for epoch in range(n_epochs):
            old_centroids = self.centroids
            distance = self.compute_distance(features, old_centroids)
            clusters = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(features, clusters)

            # check for convergence
            if np.all(old_centroids == self.centroids):
                break

            print(f'epoch { epoch + 1 } finished')

    def run(self, features, targets):
        """
        Run

        This method will run the given features through the K-Means classifier
        and will return the predicted targets of each feature according to the
        current clusters.

        Args:
            features (np.array): the features matrix
            targets (np.array): the targets matrix

        Returns:
            predicted_targets (np.array): the predicted targets of the features
            accuracy (float): the accuracy of the predicted targets
        """

        distance = self.compute_distance(features, self.centroids)
        predicted_targets = self.find_closest_cluster(distance)

        accuracy = np.sum(predicted_targets == targets.flatten()) / len(predicted_targets)
        print(f'accuracy of the K-Means classifier: { accuracy }')

        return predicted_targets


class KMeansPlusPlusClassifier():

    def __init__(self, n_clusters=1):

        # set the given number of clusters
        self.n_clusters = n_clusters

    def initialize_centroids(self, features):
        """
        Initialize Centroids

        This method will initialize the initial centroids according to the
        K-Means++ centroid initialization algorithm. This algorithm ensures that
        each initial cluster is placed appropriately.

        Args:
            features (np.array): the features matrix

        Returns:
            centroids (np.array): the initial centroids
        """

        centroids = np.zeros((self.n_clusters, features.shape[1]))

        # randomly assign the first centroid from the feature data
        centroids[0, :] = np.random.permutation(features)[0]

        # assign clusters for the remaining number of clusters
        for i in range(1, self.n_clusters):

            # find distance to closest cluster
            distance = self.compute_distance(features, centroids)
            min_distance = np.min(distance, axis=1)

            # find the max distance to the closest cluster of each feature
            max_distance_index = np.argmax(min_distance)

            # add the feature with the greatest distance to its closest cluster
            centroids[i, :] = features[max_distance_index]

        return centroids

    def compute_distance(self, features, centroids):
        """
        Compute Distance

        This method will compute the distances of all the features to the
        centroids using Euclidean distance.

        Args:
            features (np.array): the features matrix
            centroids (np.array): the centroids of each cluster

        Returns:
            distance (np.array): the distances from each feature to each cluster
        """

        distance = np.zeros((features.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(features - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)

        return distance

    def find_closest_cluster(self, distance):
        """
        Find Closest Cluster

        This method will find the closest cluster of each feature provided by
        the distance to each cluster from each feature.

        Args:
            distance (np.array): the distance matrix

        Returns:
            clusters (np.array): the closest clusters
        """

        clusters = np.argmin(distance, axis=1)

        return clusters

    def compute_centroids(self, features, clusters):
        """
        Compute Centroids

        This method will compute the centroids of each cluster.

        Args:
            features (np.array): the features matrix
            clusters (np.array): the assigned clusters of each feature

        Returns:
            centroids (np.array): the centroids
        """

        centroids = np.zeros((self.n_clusters, features.shape[1]))

        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(features[clusters == k, :], axis=0)

        return centroids

    def run(self, features, targets):
        """
        Run

        This method will run the given features through the K-Means++ classifier
        and will return the predicted targets of each feature according to the
        current clusters.

        Args:
            features (np.array): the features matrix
            targets (np.array): the targets matrix

        Returns:
            predicted_targets (np.array): the predicted targets of the features
        """

        distance = self.compute_distance(features, self.centroids)
        predicted_targets = self.find_closest_cluster(distance)

        accuracy = np.sum(predicted_targets == targets.flatten()) / len(predicted_targets)
        print(f'accuracy of the K-Means++ classifier: { accuracy }')

        return predicted_targets

    def train(self, features, n_epochs=1):
        """
        Train

        This method will train the K-Means++ Classifier with the
        provided feature data for the given number of epochs. The K-Means++
        training algorithm trains exacly as the K-Means training algorithm.

        Args:
            features (np.array): the features matrix
            n_epochs (int): the number of epochs to train
        """

        # initialize centroids
        self.centroids = self.initialize_centroids(features)

        for epoch in range(n_epochs):
            old_centroids = self.centroids
            distance = self.compute_distance(features, old_centroids)
            clusters = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(features, clusters)

            # check for convergence
            if np.all(old_centroids == self.centroids):
                break

            print(f'epoch { epoch + 1 } finished')
