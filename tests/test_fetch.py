import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, Mock

from fetch import circular_dedupe_key, classify_link_status, check_pdf_link


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


class TestLinkHealth(unittest.TestCase):
    def test_404_and_410_are_broken(self):
        self.assertEqual(classify_link_status(404), 'broken')
        self.assertEqual(classify_link_status(410), 'broken')

    def test_2xx_3xx_are_ok(self):
        self.assertEqual(classify_link_status(200), 'ok')
        self.assertEqual(classify_link_status(301), 'ok')

    def test_other_statuses_are_unknown(self):
        # 403 (bot-blocked) and 5xx are inconclusive, not confidently broken.
        self.assertEqual(classify_link_status(403), 'unknown')
        self.assertEqual(classify_link_status(500), 'unknown')
        self.assertEqual(classify_link_status(None), 'unknown')

    @patch('fetch.requests.head')
    def test_check_pdf_link_ok(self, mock_head):
        mock_head.return_value = Mock(status_code=200)
        status, http_status = check_pdf_link('https://example.com/a.pdf')
        self.assertEqual((status, http_status), ('ok', 200))

    @patch('fetch.requests.get')
    @patch('fetch.requests.head')
    def test_check_pdf_link_falls_back_to_get_on_405(self, mock_head, mock_get):
        mock_head.return_value = Mock(status_code=405)
        mock_response = Mock(status_code=200)
        mock_get.return_value = mock_response
        status, http_status = check_pdf_link('https://example.com/a.pdf')
        self.assertEqual((status, http_status), ('ok', 200))
        mock_response.close.assert_called_once()

    @patch('fetch.requests.head', side_effect=Exception("boom"))
    def test_check_pdf_link_swallows_unexpected_errors_as_unknown(self, mock_head):
        status, http_status = check_pdf_link('https://example.com/a.pdf')
        self.assertEqual((status, http_status), ('unknown', None))


if __name__ == '__main__':
    unittest.main()
