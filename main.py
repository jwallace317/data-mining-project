"""
Main Module
"""

# import necessary modules
import sys
import time
import constants
import utils


# task main
def main():
    """
    Task Main
    """

    # create unique token id dictionary
    start = time.time()
    print('creating unique token id dictionary...\n')
    document_count, token_id_dictionary, token_frequency = utils.create_token_id_dictionary(
        constants.DATASET_RELATIVE_PATH)

    print(f'total number of documents analyzed: { document_count }\n')
    end = time.time()
    elapsed_time = end - start
    print(f'elapsed time: { elapsed_time }\n')

    # create feature matrix
    start = time.time()
    print('creating full size feature matrix...\n')
    feature_matrix = utils.create_feature_matrix(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_id_dictionary)

    feature_matrix_size = sys.getsizeof(feature_matrix)
    print(f'feature matrix shape: { feature_matrix.shape }')
    print(f'feature matrix size: { feature_matrix_size }\n')
    end = time.time()
    elapsed_time = end - start
    print(f'elapsed time: { elapsed_time }\n')

    # create pruned feature matrix
    start = time.time()
    print('creating pruned feature matrix...\n')
    pruned_feature_matrix = utils.create_pruned_feature_matrix(
        constants.DATASET_RELATIVE_PATH,
        document_count,
        token_frequency,
        constants.PRUNED_SIZE)

    pruned_feature_matrix_size = sys.getsizeof(pruned_feature_matrix)
    print(f'pruned feature matrix shape: { pruned_feature_matrix.shape }')
    print(f'pruned feature matrix size: { pruned_feature_matrix_size }\n')
    end = time.time()
    elapsed_time = end - start
    print(f'elapsed time: { elapsed_time }\n')

    # create target vector
    start = time.time()
    print('creating target vector...\n')
    target_vector = utils.create_target_vector(
        constants.DATASET_RELATIVE_PATH,
        document_count)

    target_vector_size = sys.getsizeof(target_vector)
    print(f'target vector shape: { target_vector.shape }')
    print(f'target vector size: { target_vector_size }\n')
    end = time.time()
    elapsed_time = end - start
    print(f'elapsed time: { elapsed_time }\n')


if __name__ == '__main__':
    main()
