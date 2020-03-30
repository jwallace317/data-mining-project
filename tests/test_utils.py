"""
Utility Methods Test Module
"""

# import necessary modules
import unittest
import utils


class PreprocessTokenTest(unittest.TestCase):

    def test_preprocess_numbers_token(self):
        TEST_TOKEN = '123456789'

        token = utils.preprocess(TEST_TOKEN)

        self.assertEqual('!', token)

    def test_preprocess_remove_characters_token(self):
        TEST_TOKEN = 't<e>s/t^'

        token = utils.preprocess(TEST_TOKEN)

        self.assertEqual('test', token)

    def test_preprocess_remove_token(self):
        TEST_TOKEN = '<from>'

        token = utils.preprocess(TEST_TOKEN)
        self.assertEqual('!', token)
