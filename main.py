"""
Main Module
"""

# import necessary modules
import os
import sys
import time
from sklearn.model_selection import train_test_split
import constants
import utils
from naive_bayes_classifier import NaiveBayesClassifier
import numpy as np


# task main
def main():
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


def main1():
    document_contents = open(
        '/Users/jameswallace/Documents/osu/sp20/cse5243/document_classification/20news-train/alt.atheism/49960', 'rb').read().decode('utf-8', 'ignore')

    id_count = 0
    token_id = {}
    token_count = {}
    for count, token in enumerate(utils.tokenize(document_contents, constants.DELIMITERS)):
        token = utils.preprocess(token)

        if token != '!':
            if token not in token_id:
                token_id[token] = id_count
                token_count[token] = 1
                id_count += 1
            else:
                token_count[token] += 1

    print(f'total number of tokens parsed: { count }')
    print(f'total number of distinct tokens: { len(token_id) }')
    print(f'token id dictionary: { token_id }')
    print(f'token count dictionary: { token_count }')

    feature = np.zeros((1, len(token_id)))

    for token in utils.tokenize(document_contents, constants.DELIMITERS):
        token = utils.preprocess(token)

        if token != '!':
            feature[0, token_id[token]] += 1

    print(f'feature matrix: { feature }')


def main2():
    token_id, token_count, document_count = utils.create_token_ids(
        constants.DATASET_RELATIVE_PATH)

    features = utils.create_pruned_features(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_count,
        constants.PRUNED_SIZE)

    targets = utils.create_targets(
        constants.DATASET_RELATIVE_PATH,
        document_count)

    # nb_classifier = NaiveBayesClassifier(
    #     constants.DATASET_RELATIVE_PATH,
    #     token_id,
    #     token_count)
    #
    # train_features, test_features, train_targets, test_targets = train_test_split(
    #     features,
    #     targets,
    #     test_size=0.001)


if __name__ == '__main__':
    main2()
