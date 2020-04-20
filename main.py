"""
Main Module
"""

# import necessary modules
import time

import numpy as np
from sklearn.model_selection import train_test_split

import constants
import utils
from k_means_classifier import KMeansClassifier, KMeansPlusPlusClassifier


# task main
def main():
    """
    Task Main
    """

    # create token ids both pruned and unpruned
    start = time.time()
    print('CREATING UNIQUE TOKEN ID DICTIONARY\n')

    token_ids, pruned_token_ids, document_count = utils.create_token_ids(
        constants.DATASET_RELATIVE_PATH)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create the unpruned features and targets matrices
    start = time.time()
    print('CREATING UNPRUNED FEATURES AND TARGETS MATRICES\n')

    features, targets = utils.create_features_and_targets(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_ids)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create the pruned features and targets matrices
    start = time.time()
    print('CREATING PRUNED FEATURES AND TARGETS MATRICES\n')

    pruned_features, pruned_targets = utils.create_pruned_features_and_targets(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        pruned_token_ids)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create train and test set splits for pruned and unpruned data
    start = time.time()
    print('CREATING TRAIN AND TEST SET SPLITS FOR PRUNED AND UNPRUNED DATA')

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.05)
    train_pruned_features, test_pruned_features, train_pruned_targets, test_pruned_targets = train_test_split(pruned_features, pruned_targets, test_size=0.2)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # initialize the K-Means classifier
    start = time.time()
    print('INITIALIZING K-MEANS CLASSIFIER')

    k_means = KMeansClassifier(n_clusters=20)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # test the unpruned K-Means classifier
    start = time.time()
    print('TRAIN AND TEST THE UNPRUNED K-MEANS CLASSIFIER')

    k_means.train(test_features, n_epochs=10)
    predicted_targets = k_means.run(test_features, test_targets)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # test the pruned K-Means classifier
    start = time.time()
    print('TRAIN AND TEST THE PRUNED K-MEANS CLASSIFIER')

    k_means.train(train_pruned_features, n_epochs=10)
    predicted_targets = k_means.run(test_pruned_features, test_pruned_targets)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # initialize the K-Means++ classifier
    start = time.time()
    print('INITIALIZE THE K-MEANS++ CLASSIFIER')

    k_means_plus_plus = KMeansPlusPlusClassifier(n_clusters=20)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # test the unpruned K-Means++ classifier
    start = time.time()
    print('TRAIN AND TEST THE UNPRUNED K-MEANS++ CLASSIFIER')

    k_means_plus_plus.train(test_features, n_epochs=10)
    predicted_targets = k_means_plus_plus.run(test_features, test_targets)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # test the pruned K-Means++ classifier
    start = time.time()
    print('TRAIN AND TEST THE PRUNED K-MEANS++ CLASSIFIER')

    k_means_plus_plus.train(train_pruned_features, n_epochs=10)
    predicted_targets = k_means_plus_plus.run(test_pruned_features, test_pruned_targets)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')


if __name__ == '__main__':
    main()
