"""
Utility Methods Test Module
"""

# import necessary modules
import unittest
import utils
from tests import constants


class PreprocessTokenTest(unittest.TestCase):
    """
    Preprocessing Tests
    """

    def test_preprocess_token_with_numbers(self):
        """
        Test Preprocess Token with Numbers
        """

        token = utils.preprocess(constants.TOKEN_NUMBERS)

        self.assertEqual(constants.INVALID_TOKEN, token)

    def test_preprocess_token_with_unwanted_characters(self):
        """
        Test Preprocess Token with Unwanted Characters
        """

        token = utils.preprocess(constants.TOKEN_UNWANTED_CHARACTERS)

        self.assertEqual(
            constants.TOKEN_UNWANTED_CHARACTERS_REMOVED,
            token)

    def test_preprocess_token_stop_word(self):
        """
        Test Preprocess Token Stop Word
        """

        token = utils.preprocess(constants.TOKEN_STOP_WORD)

        self.assertEqual(constants.INVALID_TOKEN, token)

    def test_preprocess_token_no_vowels(self):
        """
        Test Preprocess Token with No Vowels
        """

        token = utils.preprocess(constants.TOKEN_NO_VOWELS)

        self.assertEqual(constants.INVALID_TOKEN, token)

    def test_preprocess_token_one_letter(self):
        """
        Test Preprocess Token with One Letter
        """

        token = utils.preprocess(constants.TOKEN_ONE_LETTER)

        self.assertEqual(constants.INVALID_TOKEN, token)
