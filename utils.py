"""
Utility Methods Module
"""

# import necessary modules
import os
import re
import constants


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
    token-id dictionary. This method will also write the contents of the
    unique token-id dictionary to a file.
    """

    id_count = 0
    token_id_dictionary = {}
    for topic_directory in os.listdir(path):
        topic_path = os.path.join(path, topic_directory)

        for file_name in os.listdir(topic_path):
            file_path = os.path.join(topic_path, file_name)

            file_contents = open(file_path, 'rb').read().decode(
                'utf-8', 'ignore')

            for token in tokenize(file_contents, constants.DELIMITERS):
                token = preprocess(token)

                if token != '!' and token not in token_id_dictionary:
                    token_id_dictionary[token] = id_count
                    id_count += 1

    with open('token_id_dictionary.txt', 'w') as f:
        for token, id_count in token_id_dictionary.items():
            f.write(f'{token}: {id_count}\n')

    return token_id_dictionary
