"""
Naive Bayes Classifier Module
"""

# import necessary modules
import os
import constants
import utils


class NaiveBayesClassifier():
    """
    Naive Bayes Classifier Class
    """

    def __init__(self, path):
        print('initialize naive bayes classifier...')
        self.path = path

    def create_total_frequency_dictionary(self, token_frequency):
        """
        Create Total Frequency Dictionary

        This method writes to a file the total frequency of a given token in the
        entirety of the data set.
        """

        num_tokens = len(token_frequency)

        print('initialize total frequency dictionary...')
        print('populate total frequency dictionary...')
        total_frequency_dictionary = {}
        for token, frequency in token_frequency.items():
            total_frequency_dictionary[token] = frequency / num_tokens

        print('writing total frequency dictionary to file...')
        with open('total_frequency_dictionary.txt', 'w') as file:
            file.write('total frequency dictionary\n')
            for token, frequency in total_frequency_dictionary.items():
                file.write(f'{ token }: { frequency }\n')

        return total_frequency_dictionary

    def create_document_frequency_dictionary(self, document_token_count):
        """
        Create Document Frequency Dictionary

        This method will write to many files the local token frequency of tokens
        contained in the given document.
        """

        for topic_directory in os.listdir(self.path):
            topic_path = os.path.join(self.path, topic_directory)

            for document_name in os.listdir(topic_path):
                document_path = os.path.join(topic_path, document_name)

                document_contents = open(
                    document_path, 'rb').read().decode('utf-8', 'ignore')

                token_frequency = {}
                for token in utils.tokenize(document_contents, constants.DELIMITERS):
                    token = utils.preprocess(token)

                    if token != '!':
                        if token not in token_frequency:
                            token_frequency[token] = 0
                        else:
                            token_frequency[token] += 1

                token_freq = {}
                for token, frequency in token_frequency.items():
                    token_freq[token] = frequency / \
                        document_token_count[document_name]

                with open('document_token_probs/' + topic_directory + '/' + document_name + '.txt', 'w+') as file:
                    for token, frequency in token_freq.items():
                        file.write(f'{ token }: { frequency }\n')

    def classify(self, feature, token_id_dictionary):
        return 0
