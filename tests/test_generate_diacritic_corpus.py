import unittest

from scripts.generate_diacritic_corpus import PHRASES, SAMPLES_PER_LANGUAGE


class DiacriticCorpusTests(unittest.TestCase):
    def test_locked_languages_and_phrases_cover_non_ascii_characters(self):
        self.assertEqual(set(PHRASES), {"fr", "es", "de"})
        self.assertEqual(SAMPLES_PER_LANGUAGE, 60)
        for language, phrases in PHRASES.items():
            self.assertGreaterEqual(len(phrases), 12, language)
            self.assertTrue(any(any(ord(character) > 127 for character in phrase)
                                for phrase in phrases), language)


if __name__ == "__main__":
    unittest.main()
