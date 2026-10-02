import unittest
from unittest.mock import patch

import pandas as pd

import app


def _master():
    return pd.DataFrame(
        [
            {"コード": "7203", "銘柄名": "トヨタ自動車", "市場": "プライム"},
            {"コード": "6758", "銘柄名": "ソニーグループ", "市場": "プライム"},
            {"コード": "485A", "銘柄名": "テスト自動車", "市場": "グロース"},
        ]
    )


class CompanySearchTests(unittest.TestCase):
    def test_normalize_full_width_code(self):
        self.assertEqual(app.normalize_stock_query("７２０３"), "7203")

    @patch("app.company_master", side_effect=_master)
    def test_search_by_partial_company_name(self, _mock):
        self.assertEqual(app.company_search_candidates("トヨタ")["コード"].tolist(), ["7203"])

    @patch("app.company_master", side_effect=_master)
    def test_search_supports_letter_code(self, _mock):
        self.assertEqual(app.company_search_candidates("４８５ａ")["コード"].tolist(), ["485A"])

    @patch("app.company_master", side_effect=_master)
    def test_exact_match_is_listed_before_partial_matches(self, _mock):
        self.assertEqual(app.company_search_candidates("トヨタ自動車").iloc[0]["コード"], "7203")

    @patch("app.company_master", side_effect=_master)
    def test_missing_company_returns_empty(self, _mock):
        self.assertTrue(app.company_search_candidates("存在しない会社").empty)


if __name__ == "__main__":
    unittest.main()
