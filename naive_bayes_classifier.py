"""
Naive Bayes Classifier Module
"""

# import necessary modules
import os
import sys
import constants
import utils
import numpy as np


class NaiveBayesClassifier():
    """
    Naive Bayes Classifier Class
    """

    def __init__(self, path, token_ids):
        print('initialize naive bayes classifier...')
        self.path = path
        self.token_ids = token_ids

        print('create topic probabilities matrix')
        self.topic_probs = self.create_topic_probs()
        print(
            f'topic probabilities parity check: { np.sum(self.topic_probs) }')

        print('create token given a certain topic probabilities matrix...')
        self.token_topic_probs = self.create_token_topic_probs()

    def create_topic_probs(self):
        print('initialize topic probabilities vector...')
        topic_probs = np.zeros((20, 1), dtype='float32')

        print('populate topic probabilities vector...')
        total_document_count = 0
        for topic_index, topic_directory in enumerate(os.listdir(self.path)):
            topic_path = os.path.join(self.path, topic_directory)

            document_count = 0
            for document_name in os.listdir(topic_path):
                total_document_count += 1
                document_count += 1

            topic_probs[topic_index] = document_count

        topic_probs = np.divide(topic_probs, total_document_count)

        topic_probs_size = sys.getsizeof(topic_probs)
        print(f'topic probabilities vector shape: { topic_probs.shape }')
        print(f'topic probabilities vector size: { topic_probs_size }')

        return topic_probs

    def create_token_topic_probs(self):
        print('initialize token given a certain topic probabilities matrix...')
        token_topic_probs = np.zeros(
            (20, len(self.token_ids)),
            dtype='float32')

        print('populate token given a certain topic probabilities matrix...')
        for topic_index, topic_directory in enumerate(os.listdir(self.path)):
            topic_path = os.path.join(self.path, topic_directory)

            topic_tokens_count = 0
            topic_token_count = {}
            for document_name in os.listdir(topic_path):
                document_path = os.path.join(topic_path, document_name)

                document_contents = open(
                    document_path, 'rb').read().decode('utf-8', 'ignore')

                for token in utils.tokenize(document_contents, constants.DELIMITERS):
                    token = utils.preprocess(token)

                    if token != '!' and token in self.token_ids:
                        if token in topic_token_count:
                            topic_token_count[token] += 1
                        else:
                            topic_token_count[token] = 1
                        topic_tokens_count += 1

            for token, count in topic_token_count.items():
                token_topic_probs[topic_index, self.token_ids[token]
                                  ] = count / topic_tokens_count

        token_topic_probs_size = sys.getsizeof(token_topic_probs)
        print(
            f'token given topic probabilities matrix shape: { token_topic_probs.shape }')
        print(
            f'token given topic probabilities matrix size: { token_topic_probs_size }')

        return token_topic_probs

    def predict(self, feature):
        probs = []
        for topic_index, topic_directory in enumerate(os.listdir(constants.DATASET_RELATIVE_PATH)):
            prob = 0
            for token, count in enumerate(feature):
                if count != 0:
                    prob += self.token_topic_probs[topic_index, token]
            prob += self.topic_probs[topic_index]
            probs.append(prob)

        probs_sum = np.sum(probs)
        probs = np.divide(probs, probs_sum)
        print(f'probs sum parity check = { np.sum(probs) }')

        return np.argmax(probs)
