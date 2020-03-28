"""
Utility Methods Module
"""

# import necessary modules
import os
import re
import constants
import numpy as np


def preprocess(token):
    """
    Preprocess

    token - the string token to be preprocessed

    This method will preprocess a token and determine if a given token is a
    relevant piece of information.
    """

    token = token.lower()

    for character in token:
        if character == '@' or character in constants.NUMBERS:
            token = '!'
            break

        if character in constants.REMOVE_CHARACTERS:
            token = token.replace(character, '')

    return token


def tokenize(string, delimiters):
    """
    Tokenize

    string - string to be tokenized
    delimiters - string of delimiters to tokenize by

    This method tokenizes the given string by the given delimiters.
    """

    return re.split('[' + delimiters + ']', string)


def create_token_id_dictionary(path):
    """
    Create Token Dictionary

    path - relative path to data set

    This method scans through the provided documents and creates a unique
    token-id dictionary as well as a token frequency dictionary. This method
    will also write the contents of the unique token-id dictionary to a file.
    """

    document_count = 0
    id_count = 0
    token_id_dictionary = {}
    token_frequency = {}
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        for document_name in os.listdir(topic_path):
            document_count += 1
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!' and token not in token_id_dictionary:
                    token_id_dictionary[token] = id_count
                    id_count += 1

                if token != '!' and token not in token_frequency:
                    token_frequency[token] = 0
                elif token != '!' and token in token_frequency:
                    token_frequency[token] += 1

    with open('token_id_dictionary.txt', 'w') as f:
        f.write('token id dictionary')
        f.write(f'total number of documents tokenized: { document_count }\n')
        f.write(f'total number of distinct tokens: { id_count }\n')
        for token, id_count in token_id_dictionary.items():
            f.write(f'{ token }: { id_count }\n')

    return document_count, token_id_dictionary, token_frequency


def prune_token_id_dictionary(token_id_dictionary, token_frequency, max_tokens):
    """
    Prune Token Id Dictionary

    token_id_dictionary - the unique token id dictionary
    max_tokens - the max number of the most frequent tokens to keep

    This method will prune the given token id dictionary and remove all the
    tokens that are not within the max number of most frequent tokens. This
    method will also write to a file the most frequent tokens sorted in
    descending order.
    """

    token_frequency_sorted = sorted(
        token_frequency.items(), key=lambda x: x[1], reverse=True)
    pruned_token_frequency = token_frequency_sorted[0:max_tokens]

    with open('token_frequency_dictionary.txt', 'w') as f:
        f.write('token frequency dictionary')
        for token, frequency in token_frequency_sorted:
            f.write(f'{ token }: { frequency }\n')

    count = 0
    pruned_token_id_dictionary = {}
    for token, frequency in pruned_token_frequency:
        pruned_token_id_dictionary[token] = count
        count += 1

    return pruned_token_id_dictionary


def create_feature_matrix(path, document_count, token_id_dictionary):
    """
    Create Feature Matrix

    path - relative path to the data set
    token_id_dictionary - unique token id dictionary

    This method will create a feature matrix for the provided documents in the
    data set.
    """

    feature_matrix = np.zeros(
        (document_count, len(token_id_dictionary)),
        dtype='uint8')

    document_count = 0
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        for document_name in os.listdir(topic_path):
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!':
                    feature_matrix[document_count,
                                   token_id_dictionary[token]] += 1

            document_count += 1

    return feature_matrix


def create_pruned_feature_matrix(path, document_count, token_id_dictionary, token_frequency, pruned_size):
    """
    Create Pruned Feature Matrix

    path - relative path to the data set
    document_count - total number of documents within the data set
    token_id_dictionary - unique token id dictionary
    pruned_size - the desired pruned size of the feature matrix

    This method will create a pruned feature matrix.
    """

    pruned_feature_matrix = np.zeros(
        (document_count, pruned_size),
        dtype='uint8')

    pruned_token_id_dictionary = prune_token_id_dictionary(
        token_id_dictionary,
        token_frequency,
        pruned_size)

    document_count = 0
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        for document_name in os.listdir(topic_path):
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!' and token in pruned_token_id_dictionary:
                    pruned_feature_matrix[document_count,
                                          pruned_token_id_dictionary[token]] += 1

            document_count += 1

        return pruned_feature_matrix


def create_target_vector(path, document_count):
    """
    Create Target vector

    path - the relative path to the data set
    document_count - the total number of documents provided within the data set

    This method will create the target vector given by the provided data set.
    """

    target_vector = np.zeros((document_count, 1))

    document_count = 0
    topic_count = 0
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        for document_name in os.listdir(topic_path):
            target_vector[document_count] = topic_count
            document_count += 1

        topic_count += 1

    return target_vector
