import unittest

from home_research import activate_ranked_stock


class NavigationTest(unittest.TestCase):
    def test_home_recommendation_opens_search_page(self):
        state = {"main_menu": "ホーム"}

        activate_ranked_stock(state, "485a")

        self.assertEqual(state["main_menu"], "検索")
        self.assertEqual(
            state["active_analysis"],
            {"code": "485A", "period_name": "1年"},
        )


if __name__ == "__main__":
    unittest.main()
