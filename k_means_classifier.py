"""
K-Means Classifier Module
"""

# import necessary modules
import numpy as np


class KMeansClassifier():

    def __init__(self, n_clusters=1):
        self.n_clusters = n_clusters
        self.centroids = np.zeros(n_clusters)

    def initialize_centroids(self, features):
        centroids = np.random.permutation(features)[0:self.n_clusters]
        return centroids

    def compute_distance(self, features, centroids):
        distance = np.zeros((features.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(features - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_centroids(self, features, targets):
        centroids = np.zeros((self.n_clusters, features.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(features[targets == k, :], axis=0)
        return centroids

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def train(self, features, n_epochs=1):
        self.centroids = self.initialize_centroids(features)
        for i in range(n_epochs):
            old_centroids = self.centroids
            distance = self.compute_distance(features, old_centroids)
            clusters = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(features, clusters)

            if np.all(old_centroids == self.centroids):
                break

            print(f'epoch { i + 1 } finished')

    def find_cluster_targets(self, features, targets):
        distance = self.compute_distance(features, self.centroids)
        clusters = self.find_closest_cluster(distance)

        clusters_targets = []
        for i in range(self.n_clusters):
            cluster_targets = targets[clusters == i]

            counts = np.bincount(cluster_targets.flatten())
            clusters_targets.append(np.argmax(counts))

        self.cluster_labels = clusters_targets
        print(self.cluster_labels)

    def run(self, features):
        distance = self.compute_distance(features, self.centroids)
        predicted_targets = self.find_closest_cluster(distance)

        return predicted_targets
