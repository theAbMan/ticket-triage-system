import unittest
from src.preprocessing import clean_text

class TestPreprocessing(unittest.TestCase):
    def test_clean_text(self):
        text = "  Hello! How are YOU? :) "
        expected = "hello how are you"
        self.assertEqual(clean_text(text), expected)


if __name__ == "__main__":
    unittest.main()