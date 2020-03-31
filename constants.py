"""
Constants Module

This module contains all the constant values that are used during document
classification by various algorithms.
"""

DATASET_RELATIVE_PATH = '20news-train/'

DEFAULT_PRUNED_SIZE = 5000

DELIMITERS = ' \n-@.\t'

NUMBERS = '0123456789'

REMOVE_CHARACTERS = '<>|*():,!?\';/\\\"_[]`{}=~%^'

REMOVE_TOKENS = ['from', 'a', 'the', 'to', 'in', 'of', 'an',
                 'and', 'by', 'that', 'it', 'is', '', 'with', 'for', 'reply',
                 'were', 'are', 'its', 'such', 'very', 'when', 'i', 'be', 'or',
                 'this', 'me', 're', 'im', 'if', 'as', 'dont', 'you', 'not',
                 'was', 'but', 'they', 'there', 'what', 'can', 'your', 'has',
                 'some', 'who', 'subject', 'organization', 'will', 'would']

VOWELS = 'aeiou'
