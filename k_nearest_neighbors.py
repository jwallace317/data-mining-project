"""
K Nearest Neighbors Module
"""

# import necessary modules
import numpy as np


class KNearestNeighbors():
    """
    K Nearest Neighbors Class
    """

    def __init__(self, features, targets, k_neighbors=1):
        self.k_neighbors = k_neighbors
        self.features = features
        self.targets = targets

    def predict(self, feature):
        """
        Predict

        Args:
            feature: The input feature.

        Returns:
            The predicted target according to the K Nearest Neighbors algorithm.
        """

        distances = np.array([])

        for neighbor in self.features:
            distances = np.append(
                distances, np.linalg.norm(feature - neighbor))

        idx = np.argpartition(distances, self.k_neighbors)

        k_nearest_neighbors = self.targets[idx[0:self.k_neighbors]]
        values, counts = np.unique(k_nearest_neighbors, return_counts=True)

        idx = np.argmax(counts)

        return values[idx]
