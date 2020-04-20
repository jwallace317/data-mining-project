"""
Main Module
"""

# import necessary modules
import time

import numpy as np
from sklearn.model_selection import train_test_split

import constants
import utils
from k_means_classifier import KMeansClassifier


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

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)
    train_pruned_features, test_pruned_features, train_pruned_targets, test_pruned_targets = train_test_split(pruned_features, pruned_targets, test_size=0.2)

    print(f'train features shape: {train_features.shape}')
    print(f'train targets shape: {train_targets.shape}')
    print(f'test features shape: {test_features.shape}')
    print(f'test targets shape: {test_targets.shape}')
    print(f'train pruned features shape: {train_pruned_features.shape}')
    print(f'train pruned targets shape: {train_pruned_targets.shape}')
    print(f'test pruned features shape: {test_pruned_features.shape}')
    print(f'test pruned targets shape: {test_pruned_targets.shape}')

    k_means_classifier = KMeansClassifier(n_clusters=20)

    k_means_classifier.train(test_pruned_features, n_epochs=100)

    k_means_classifier.find_cluster_targets(test_pruned_features, test_pruned_targets)

    predicted_targets = k_means_classifier.run(test_pruned_features)

    correct = 0
    for prediction, actual in zip(predicted_targets, test_pruned_targets):
        print(prediction)
        print(actual)

        if prediction == actual:
            print('CORRECT')
            correct += 1

    print(correct)
    print(f'accurcy: { correct / len(predicted_targets) }')


if __name__ == '__main__':
    main()
