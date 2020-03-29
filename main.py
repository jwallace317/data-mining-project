"""
Main Module
"""

# import necessary modules
import sys
import time
import constants
import utils
from naive_bayes_classifier import NaiveBayesClassifier


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
    nb_classifier = NaiveBayesClassifier(constants.DATASET_RELATIVE_PATH)

    end = time.time()
    elapsed_time = end - start
    print(f'\nELAPSED TIME: { elapsed_time }\n')


if __name__ == '__main__':
    main()
