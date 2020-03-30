"""
Main Module
"""

# import necessary modules
import time
from sklearn.model_selection import train_test_split
import numpy as np
import constants
import utils
from naive_bayes_classifier import NaiveBayesClassifier
from sklearn.neighbors import KNeighborsClassifier
from k_nearest_neighbors import KNearestNeighbors


# task main
def main():

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

    # create unpruned naive bayes classifier
    start = time.time()
    print('CREATING UNPRUNED NAIVE BAYES CLASSIFIER\n')

    nb_classifier = NaiveBayesClassifier(
        constants.DATASET_RELATIVE_PATH,
        token_ids)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create pruned naive bayes classifier
    start = time.time()
    print('CREATING PRUNED NAIVE BAYES CLASSIFIER\n')

    pruned_nb_classifier = NaiveBayesClassifier(
        constants.DATASET_RELATIVE_PATH,
        pruned_token_ids)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create train and test sets for the unpruned features and targets matrices
    start = time.time()
    print('CREATING TRAIN AND TEST SETS FOR UNPRUNED DATA\n')

    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=0.0025)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create train and test sets for the pruned features and targets matrices
    start = time.time()
    print('CREATING TRAIN AND TEST SETS FOR PRUNED DATA\n')

    pruned_train_features, pruned_test_features, pruned_train_targets, pruned_test_targets = train_test_split(
        pruned_features,
        pruned_targets,
        test_size=0.0025)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    print(f'test size = { len(test_targets) }')
    print(f'pruned test size = { len(pruned_test_targets) }')

    correct = 0
    for feature, target in zip(test_features, test_targets):
        prediction = nb_classifier.predict(feature)

        if prediction == target:
            correct += 1

    pruned_correct = 0
    for feature, target in zip(pruned_test_features, pruned_test_targets):
        prediction = pruned_nb_classifier.predict(feature)

        if prediction == target:
            pruned_correct += 1

    print(
        f'accuracy of naive bayes classifier = { correct / len(test_targets)}')
    print(
        f'accuracy of pruned naive bayes classifier = { pruned_correct / len(pruned_test_targets) }')

    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=0.1)

    pruned_train_features, pruned_test_features, pruned_train_targets, pruned_test_targets = train_test_split(
        pruned_features,
        pruned_targets,
        test_size=0.1)

    knn_classifier = KNearestNeighbors(
        pruned_test_features,
        pruned_test_targets,
        k_neighbors=20)

    print(f'sample size = { len(pruned_test_targets) }')

    correct = 0
    for feature, target in zip(pruned_test_features, pruned_test_targets):
        prediction = knn_classifier.predict(feature)

        if prediction == target:
            correct += 1

    print(f'accuracy = { correct / len(pruned_test_targets) }')


if __name__ == '__main__':
    main()
