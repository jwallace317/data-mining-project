"""
Main Module
"""

# import necessary modules
import os
import re
import numpy as np
import constants
from kmeans import Kmeans
import utils


def preprocess(token):
    """
    preprocess token
    """

    token = token.lower()

    for character in token:
        if character == '@' or character in constants.NUMBERS:
            return ['!']  # do not use token

        if character in constants.DELIMITERS:
            token = token.replace(character, '')

    tokens = []
    if '\n' in token or '-' in token:
        tokens = re.split('[\n-]', token)
    else:
        tokens.append(token)

    return tokens


# task main
def main():
    """
    Task Main
    """

    # create unique token id dictionary
    token_id_dictionary = utils.create_token_id_dictionary(
        constants.DATASET_RELATIVE_PATH)

    # create feature matrix

    # create pruned feature matrix

    # # count variables
    # file_count = 0
    # id_count = 0
    # token_count = 0
    # token_dictionary = {}
    # token_frequency = {}
    #
    # # iterate through the topic subdirectories
    # for topic_directory in os.listdir(constants.DATASET_RELATIVE_PATH):
    #     topic_path = os.path.join(
    #         constants.DATASET_RELATIVE_PATH, topic_directory)
    #
    #     # iterate through the files within each topic subdirectory
    #     for file_name in os.listdir(topic_path):
    #         file_count += 1
    #         file_path = os.path.join(topic_path, file_name)
    #
    #         # read in file contents
    #         file_contents = open(file_path, 'rb').read().decode(
    #             'utf-8', 'ignore')
    #
    #         # tokenize the file
    #         for token in file_contents.split(' '):
    #             token_count += 1
    #
    #             # preprocess the token
    #             tokens = preprocess(token)
    #
    #             # populate the word-id dictionary
    #             for token in tokens:
    #                 if token not in token_dictionary and token != '!':
    #                     token_dictionary[token] = id_count
    #                     token_frequency[token] = 1
    #                     id_count += 1
    #                 elif token in token_dictionary:
    #                     token_frequency[token] += 1
    #
    # token_frequency_sorted = sorted(
    #     token_frequency.items(), key=lambda x: x[1], reverse=True)
    # token_frequency_pruned = token_frequency_sorted[0:constants.PRUNED_SIZE]
    # token_dictionary_pruned = {}
    # count = 0
    # for token, frequency in token_frequency_pruned:
    #     token_dictionary_pruned[token] = count
    #     count += 1
    #
    # print(f'token dictionary pruned size: { len(token_dictionary_pruned) }')
    # with open('token_dictionary.txt', 'w') as f:
    #     for token, id_count in token_dictionary.items():
    #         f.write(f'{ token }: { id_count }\n')
    #
    # with open('token_frequency.txt', 'w') as f:
    #     for token, count in token_frequency_sorted:
    #         f.write(f'{ token }: { count }\n')
    #
    # with open('token_frequency_pruned.txt', 'w') as f:
    #     for token, count in token_frequency_sorted[0:constants.PRUNED_SIZE]:
    #         f.write(f'{ token }: { count }\n')
    #
    # print(f'total number of files: { file_count }')
    # print(f'number of tokens: { token_count }')
    # print(f'number of distinct words: { id_count }')
    #
    # # create feature matrix with proper dimensions
    # features = np.zeros((file_count, id_count))
    # features_pruned = np.zeros((file_count, constants.PRUNED_SIZE))
    #
    # # create the target vector with proper dimensions
    # targets = np.zeros((file_count, 1))
    #
    # # reset file count
    # file_count = 0
    # topic_id = 0
    #
    # # iterate through the topic subdirectories
    # for topic_directory in os.listdir(constants.DATASET_RELATIVE_PATH):
    #     topic_path = os.path.join(
    #         constants.DATASET_RELATIVE_PATH, topic_directory)
    #
    #     # iterate through the files within each topic subdirectory
    #     for file_name in os.listdir(topic_path):
    #         file_path = os.path.join(topic_path, file_name)
    #
    #         # read in file contents
    #         file_contents = open(file_path, 'rb').read().decode(
    #             'utf-8', 'ignore')
    #
    #         # tokenize the file
    #         for token in file_contents.split(' '):
    #
    #             # preprocess the token
    #             tokens = preprocess(token)
    #
    #             for token in tokens:
    #                 if token != '!':
    #                     # insert into feature matrix
    #                     features[file_count, token_dictionary[token]] += 1
    #
    #                     if token in token_dictionary_pruned:
    #                         features_pruned[file_count,
    #                                         token_dictionary_pruned[token]] += 1
    #
    #         # populate the target vector
    #         targets[file_count, 0] = topic_id
    #
    #         file_count += 1
    #     topic_id += 1
    #
    # print(f'feature matrix shape = { features.shape }')
    # print(f'feature matrix = { features[1, 1:100] }\n')
    # print(f'pruned feature matrix shape = { features_pruned.shape }')
    # print(f'pruned feature matrix ={ features_pruned[1, 1:100] }\n')
    # print(f'target vector shape = { targets.shape }')
    # print(f'target vector = { targets[1:100, 0] }\n')
    #
    # kmeans = Kmeans(n_clusters=20, max_epochs=10, intra_cluster_variance=False)
    # kmeans.train(features_pruned)
    # print(kmeans.centroids)
    # kmeans.determine_labels(features_pruned, targets)

    return 0


if __name__ == '__main__':
    main()
