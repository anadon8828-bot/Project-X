import unittest
from datetime import datetime
import pandas as pd
from research_rules import fresh_rows, valid_materials, assessment, expected_price_day
from candidate_watchlist import select_watchlist, select_by_price_band
from verified_materials import parse_page, classify
from risk_engine import make_trade_plan


class ResearchIntegrity(unittest.TestCase):
    def test_calendar(self):
        self.assertEqual(expected_price_day(pd.Timestamp('2026-09-20T15:00:00+09:00')),'2026-09-18')
        self.assertEqual(expected_price_day(pd.Timestamp('2026-09-18T08:00:00+09:00')),'2026-09-17')
        self.assertEqual(expected_price_day(pd.Timestamp('2026-09-18T10:00:00+09:00')),'2026-09-17')
        self.assertEqual(expected_price_day(pd.Timestamp('2026-09-18T16:00:00+09:00')),'2026-09-18')
    def test_freshness(self):
        # Tokyo daily bars are not treated as complete until after the close.
        now=pd.Timestamp('2026-09-18T16:00:00+09:00')
        rows=pd.DataFrame([{'株価基準日':'2026-09-18','取得日時':s} for s in ['2026-09-18T14:40:00+09:00','2026-09-18T14:00:00+09:00','2026-09-18T16:00:00+09:00','bad']])
        self.assertEqual(len(fresh_rows(rows,now)),3)
        self.assertEqual(len(fresh_rows(rows,pd.Timestamp('2026-09-19T15:00:00+09:00'))),3)
        delayed=pd.DataFrame([{'株価基準日':'2026-09-17','取得日時':'2026-09-18T14:00:00+09:00'}])
        older=pd.DataFrame([{'株価基準日':'2026-09-16','取得日時':'2026-09-18T14:00:00+09:00'}])
        self.assertEqual(len(fresh_rows(delayed,now)),1)
        self.assertTrue(fresh_rows(older,now).empty)
        self.assertTrue(fresh_rows(pd.DataFrame(),now).empty)

    def test_price_band_selection(self):
        prices=[500,800,1500,2500,4500,7000,12000,18000,2200,5200,900]
        data=pd.DataFrame([
            {'コード':str(1000+i),'終値':price,'テクニカル一致数':4-i%4,'平均売買代金(百万円)':1000-i}
            for i,price in enumerate(prices)
        ])
        rows=select_by_price_band(data,limit=10)
        self.assertEqual(len(rows),10)
        self.assertEqual(set(rows['価格帯']),{'1,000円未満','1,000〜3,000円','3,000〜10,000円','10,000円以上'})

    def test_material_gates(self):
        self.assertFalse(classify('自己株式取得状況')[1])
        self.assertFalse(classify('上方修正の訂正')[1])
        self.assertFalse(classify('業績予想の修正')[1])
        self.assertTrue(classify('自己株式取得に係る事項の決定')[1])
        self.assertFalse(classify('増配および業績下方修正')[1])

    def test_code_source_time(self):
        html='<table id="main-list-table"><tr><td class="kjCode">485A0</td><td class="kjTime">14:00</td><td class="kjTitle"><a href="test.pdf">増配</a></td></tr></table>'
        now=datetime.fromisoformat('2026-09-18T15:00:00+09:00')
        rows,_=parse_page(html,'2026-09-18',{'485A'},now)
        self.assertEqual(rows[0]['code'],'485A')
        self.assertEqual(parse_page(html,'2026-09-18',{'7203'},now)[0],[])
        self.assertEqual(parse_page(html,'2026-09-19',{'485A'},now)[0],[])
        self.assertEqual(parse_page(html.replace('test.pdf','https://evil.example/test.pdf'),'2026-09-18',{'485A'},now)[0],[])

    def test_same_decision_and_low_ai(self):
        news=[{'code':'485A','title':'増配','positive':True}]
        data=pd.DataFrame([{'コード':'485A','テクニカル一致数':0,'AI上昇確率(%)':1},{'コード':'7203','テクニカル一致数':4,'AI上昇確率(%)':1}])
        rows=select_watchlist(data,materials=news)
        self.assertEqual(set(rows['コード']),{'485A','7203'})
        self.assertEqual(rows.iloc[0]['候補理由'],assessment(data.iloc[0],news)['候補理由'])
        self.assertNotEqual(rows.iloc[0]['現在の判断'],'買い条件成立')

    def test_news_expiry(self):
        now=pd.Timestamp('2026-09-18T15:00:00+09:00')
        self.assertEqual(valid_materials({'checked':'2026-09-18T14:00:00+09:00','records':[{'published':now.isoformat()}]},now),[])

    def test_unapproved_rule_never_emits_buy_candidate(self):
        plan=make_trade_plan(
            price=1000,atr=20,ai_probability=.90,probability_2pct=.90,
            expected_return=.10,technical_score=4,capital_yen=1_000_000,
        )
        self.assertEqual(plan.action,'研究候補（未採用）')
        self.assertNotIn('買い候補',plan.action)

if __name__=='__main__':
    unittest.main()
