import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fetch import circular_dedupe_key


class TestCircularDedupeKey(unittest.TestCase):
    def test_same_circular_no_and_date_match_regardless_of_case_and_spacing(self):
        a = {"circular_no": "Ho.No.13/51/2008-NATRSS", "date": "02/04/2009", "title": "NATRSS Training"}
        b = {"circular_no": "  ho.no.13/51/2008-natrss  ", "date": "02/04/2009", "title": "Different title text"}
        self.assertEqual(circular_dedupe_key(a), circular_dedupe_key(b))

    def test_different_circular_no_gives_different_key(self):
        a = {"circular_no": "A/1/2020", "date": "01/01/2020", "title": "X"}
        b = {"circular_no": "A/2/2020", "date": "01/01/2020", "title": "X"}
        self.assertNotEqual(circular_dedupe_key(a), circular_dedupe_key(b))

    def test_missing_circular_no_falls_back_to_title_and_date(self):
        a = {"circular_no": "", "date": "01/01/2020", "title": "Untitled Notice"}
        b = {"circular_no": None, "date": "01/01/2020", "title": "  untitled   notice  "}
        self.assertEqual(circular_dedupe_key(a), circular_dedupe_key(b))
        c = {"circular_no": "", "date": "01/01/2020", "title": "A different notice"}
        self.assertNotEqual(circular_dedupe_key(a), circular_dedupe_key(c))


if __name__ == '__main__':
    unittest.main()
