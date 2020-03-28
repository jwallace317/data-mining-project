"""
Utility Methods Test Module
"""

# import necessary modules
import unittest
import utils


class PreprocessTokenTest(unittest.TestCase):

    def test_preprocess_email_token(self):
        TEST_TOKEN = 'email@email.com'

        token = utils.preprocess(TEST_TOKEN)

        self.assertEqual('!', token)
