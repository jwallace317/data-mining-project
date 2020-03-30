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
def main5():
    """
    Task Main
    """

    # create unique token id dictionary
    start = time.time()
    print('CREATING UNIQUE TOKEN ID DICTIONARY\n')
    token_id_dictionary, token_count, document_count = utils.create_token_id_dictionary(
        constants.DATASET_RELATIVE_PATH)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create feature matrix
    start = time.time()
    print('CREATING FULL SIZE FEATURE MATRIX\n')
    feature_matrix = utils.create_feature_matrix(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_id_dictionary)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create pruned feature matrix
    start = time.time()
    print('CREATING PRUNED FEATURE MATRIX\n')
    pruned_feature_matrix = utils.create_pruned_feature_matrix(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_count,
        constants.PRUNED_SIZE)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create target vector
    start = time.time()
    print('CREATING TARGET VECTOR\n')
    target_vector = utils.create_target_vector(
        constants.DATASET_RELATIVE_PATH,
        document_count)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # create naive bayes classifier
    start = time.time()
    print('CREATE NAIVE BAYES CLASSIFIER\n')
    nb_classifier = NaiveBayesClassifier(
        constants.DATASET_RELATIVE_PATH,
        token_id_dictionary,
        token_count)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')

    # classify documents with naive bayes classifier
    start = time.time()
    print('CLASSIFY DOCUMENTS WITH NAIVE BAYES CLASSIFIER\n')
    train_features, test_features, train_targets, test_targets = train_test_split(
        feature_matrix,
        target_vector,
        test_size=0.001)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')


def main():

    token_ids, pruned_token_ids, document_count = utils.create_token_ids(
        constants.DATASET_RELATIVE_PATH)

    features, targets = utils.create_features_and_targets(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_ids)

    pruned_features, pruned_targets = utils.create_pruned_features_and_targets(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        pruned_token_ids)

    # nb_classifier = NaiveBayesClassifier(
    #     constants.DATASET_RELATIVE_PATH,
    #     token_ids)
    #
    # pruned_nb_classifier = NaiveBayesClassifier(
    #     constants.DATASET_RELATIVE_PATH,
    #     pruned_token_ids)
    #
    # train_features, test_features, train_targets, test_targets = train_test_split(
    #     features,
    #     targets,
    #     test_size=0.0025)
    #
    # pruned_train_features, pruned_test_features, pruned_train_targets, pruned_test_targets = train_test_split(
    #     pruned_features,
    #     pruned_targets,
    #     test_size=0.0025)
    #
    # print(f'test size = { len(test_targets) }')
    # print(f'pruned test size = { len(pruned_test_targets) }')
    #
    # correct = 0
    # for feature, target in zip(test_features, test_targets):
    #     prediction = nb_classifier.predict(feature)
    #
    #     if prediction == target:
    #         correct += 1
    #
    # pruned_correct = 0
    # for feature, target in zip(pruned_test_features, pruned_test_targets):
    #     prediction = pruned_nb_classifier.predict(feature)
    #
    #     if prediction == target:
    #         pruned_correct += 1
    #
    # print(
    #     f'accuracy of naive bayes classifier = { correct / len(test_targets)}')
    # print(
    #     f'accuracy of pruned naive bayes classifier = { pruned_correct / len(pruned_test_targets) }')

    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=0.1)

    pruned_train_features, pruned_test_features, pruned_train_targets, pruned_test_targets = train_test_split(
        pruned_features,
        pruned_targets,
        test_size=0.1)

    knn_classifier = KNearestNeighbors(
        50,
        pruned_test_features,
        pruned_test_targets)

    print(f'sample size = { len(pruned_test_targets) }')

    correct = 0
    for feature, target in zip(pruned_test_features, pruned_test_targets):
        prediction = knn_classifier.predict(feature)

        if prediction == target:
            correct += 1

    print(f'accuracy = { correct / len(pruned_test_targets) }')


if __name__ == '__main__':
    main()
