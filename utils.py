"""
Utility Methods Module
"""

# import necessary modules
import os
import re
import sys
import numpy as np
import constants


def preprocess(token):
    """
    Preprocess

    token - the string token to be preprocessed

    This method will preprocess a token and determine if a given token is a
    relevant piece of information.
    """

    token = token.lower()

    vowels = False
    for character in token:
        if character in constants.NUMBERS:
            token = '!'
            break

        if character in constants.REMOVE_CHARACTERS:
            token = token.replace(character, '')

        if character in constants.VOWELS:
            vowels = True

    if len(token) <= 2 or token in constants.REMOVE_TOKENS or not vowels:
        token = '!'

    return token


def tokenize(string, delimiters):
    """
    Tokenize

    string - string to be tokenized
    delimiters - string of delimiters to tokenize by

    This method tokenizes the given string by the given delimiters.
    """

    return re.split('[' + delimiters + ']', string)


def create_token_ids(path, pruned_size=constants.DEFAULT_PRUNED_SIZE):
    """
    Create Token Id Dictionary

    path - relative path to data set

    This method scans through the provided documents and creates a unique
    token-id dictionary as well as a token frequency dictionary. This method
    will also write the contents of the unique token-id dictionary to a file.
    """

    print('initialize token id dictionary...')
    print('populate token id dictionary...')
    document_count = 0
    id_count = 0
    token_ids = {}
    token_count = {}
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        for document_index, document_name in enumerate(os.listdir(topic_path)):
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!':
                    if token not in token_ids:
                        token_ids[token] = id_count
                        id_count += 1

                    if token in token_count:
                        token_count[token] += 1
                    else:
                        token_count[token] = 1
        document_count += document_index + 1

    print(f'{ len(token_ids) } tokens parsed...')
    print(f'{ document_count } documents analyzed...')
    print('writing token id dictionary to file...')
    with open('token_id_dictionary.csv', 'w+') as file:
        for token, id_count in token_ids.items():
            file.write(f'{ token }, { id_count }\n')

    token_count = sorted(token_count.items(), key=lambda x: x[1], reverse=True)

    print('writing token count to file...')
    with open('token_count.csv', 'w+') as file:
        for token, count in token_count:
            file.write(f'{ token }, { count }\n')

    pruned_token_ids = prune_token_ids(token_count, pruned_size)

    return token_ids, pruned_token_ids, document_count


def prune_token_ids(token_count, size):
    """
    Prune Token Id Dictionary

    token_id_dictionary - the unique token id dictionary
    max_tokens - the max number of the most frequent tokens to keep

    This method will prune the given token id dictionary and remove all the
    tokens that are not within the max number of most frequent tokens. This
    method will also write to a file the most frequent tokens sorted in
    descending order.
    """

    pruned_token_count = token_count[0:size]

    print('populate pruned token id dictionary...')
    id_count = 0
    pruned_token_ids = {}
    for token, count in pruned_token_count:
        pruned_token_ids[token] = id_count
        id_count += 1

    return pruned_token_ids


def create_features_and_targets(path, document_count, token_ids):
    """
    Create Feature Matrix

    path - relative path to the data set
    token_ids - unique token id dictionary

    This method will create a feature matrix for the provided documents in the
    data set.
    """

    print('initialize features matrix...')
    features = np.zeros((document_count, len(token_ids)), dtype='uint8')

    print('initialize targets matrix...')
    targets = np.zeros((document_count, 1), dtype='uint8')

    print('populate features and targets matrices...')
    target_topic_dict = {}
    document_count = 0
    for topic_index, topic_directory in enumerate(os.listdir(path)):
        topic_path = os.path.join(path, topic_directory)

        target_topic_dict[topic_directory] = topic_index

        for document_name in os.listdir(topic_path):
            targets[document_count, 0] = topic_index
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!':
                    features[document_count, token_ids[token]] += 1
            document_count += 1

    features_size = sys.getsizeof(features)
    print(f'features shape: { features.shape }')
    print(f'features size: { features_size }')

    targets_size = sys.getsizeof(targets)
    print(f'targets shape: { targets.shape }')
    print(f'targets size: { targets_size }')

    print('writing target topic dictionary to file...')
    with open('target_topic_dictionary.csv', 'w+') as file:
        for topic, target in target_topic_dict.items():
            file.write(f'{ topic }, { target }\n')

    return features, targets


def create_pruned_features_and_targets(path, document_count, pruned_token_ids):
    """
    Create Pruned Feature Matrix

    path - relative path to the data set
    document_count - total number of documents within the data set
    token_id_dictionary - unique token id dictionary
    pruned_size - the desired pruned size of the feature matrix

    This method will create a pruned feature matrix.
    """

    size = len(pruned_token_ids)

    print('initialize pruned feature matrix...')
    pruned_features = np.zeros((document_count, size), dtype='uint8')

    print('initialize pruned target vector...')
    pruned_targets = np.zeros((document_count, 1), dtype='uint8')

    print('populate pruned feature matrix...')
    document_count = 0
    for topic_index, topic_directory in enumerate(os.listdir(path)):
        topic_path = os.path.join(path, topic_directory)

        for document_name in os.listdir(topic_path):
            pruned_targets[document_count] = topic_index
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!' and token in pruned_token_ids:
                    pruned_features[document_count,
                                    pruned_token_ids[token]] += 1
            document_count += 1

    pruned_features_size = sys.getsizeof(pruned_features)
    print(f'pruned feature matrix shape: { pruned_features.shape }')
    print(f'pruned feature matrix size: { pruned_features_size }')

    pruned_targets_size = sys.getsizeof(pruned_targets)
    print(f'pruned targets shape: { pruned_targets.shape }')
    print(f'pruned targets size: { pruned_targets_size }')

    return pruned_features, pruned_targets


def create_targets(path, document_count):
    """
    Create Target vector

    path - the relative path to the data set
    document_count - the total number of documents provided within the data set

    This method will create the target vector given by the provided data set.
    """

    print('initialize target vector...')
    target_vector = np.zeros((document_count, 1))

    print('populate target vector...')
    target_topic = {}
    document_count = 0
    for topic_index, topic_directory in enumerate(os.listdir(path)):
        target_topic[topic_directory] = topic_index
        topic_path = os.path.join(path, topic_directory)

        for document_name in os.listdir(topic_path):
            target_vector[document_count] = topic_index
            document_count += 1

    with open('target_dictionary.csv', 'w+') as file:
        for topic, target in target_topic.items():
            file.write(f'{ topic }, { target }\n')

    target_vector_size = sys.getsizeof(target_vector)
    print(f'target vector shape: { target_vector.shape }')
    print(f'target vector size: { target_vector_size }')

    return target_vector


def create_token_topic_probs(path, token_ids):
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        token_count = 0
        topic_token_count = {}
        for document_name in os.listdir(topic_path):
            document_path = os.path.join(topic_path, document_name)

            document_contents = open(
                document_path, 'rb').read().decode('utf-8', 'ignore')

            for token in tokenize(document_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!':
                    token_count += 1
                    if token in topic_token_count:
                        topic_token_count[token] += 1

                    else:
                        topic_token_count[token] = 1

        with open('test.csv', 'w+') as file:
            file.write(f'topic: { topic_directory }\n')
            file.write(
                f'total number of distinct tokens: { len(topic_token_count)}\n')
            file.write(f'total number of tokens: { token_count }\n')
            for token, id in token_ids.items():
                if token in topic_token_count:
                    file.write(f'{ topic_token_count[token] }\n')
                else:
                    file.write(f'0\n')

        input()
