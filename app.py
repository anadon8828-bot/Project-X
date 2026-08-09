import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from technical import calculate_indicators
from ai import AIEngine
from ranking import create_ranking
from portfolio import calculate_portfolio
from backtest import run_backtest
from market import get_market_score
from history import save_prediction, show_history
from stock_results import show_stock_results
from analysis import calculate_analysis
from charts import create_chart
from signals import calculate_rebound_score, create_signal, create_decision
from dashboard import (
    show_ai_judgment,
    show_dashboard,
    show_ai_diagnosis
)
from simulation import show_simulation
from score import calculate_score


# =========================
# 設定
# =========================

st.set_page_config(
    page_title="Project X V3",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.title(
    "🚀 Project X Version 3.0"
)

st.subheader(
    "AI株価分析システム"
)


# =========================
# AI読み込み
# =========================

try:

    ai = AIEngine()

except Exception as e:

    st.error(
        str(e)
    )

    st.stop()



# =========================
# 銘柄入力
# =========================

code = st.text_input(
    "銘柄コード",
    "7203"
)


period = st.selectbox(
    "分析期間",
    [
        "6mo",
        "1y",
        "3y",
        "5y"
    ],
    index=1
)

# =========================
# 分析結果保存用
# =========================

data = None
latest = None
probability = 0
signal = ""
final_score = 0
rebound_score = 0


# =========================
# 分析
# =========================

if st.button(
    "🚀 分析開始"
):


    data = yf.Ticker(
        code + ".T"
    ).history(
        period=period
    )


    if data.empty:

        st.error(
            "データ取得失敗"
        )

        st.stop()



    # テクニカル計算

    data = calculate_indicators(
        data
    )


    latest = data.iloc[-1]


    


    # AI判定

    analysis_result = calculate_analysis(
    data,
    latest,
    ai
)

    probability = analysis_result["probability"]

    signal = analysis_result["signal"]

    rebound_score = calculate_rebound_score(
    data,
    latest
)

signal = create_signal(
    probability
)



if latest is not None:

        save_prediction(
        code,
        latest["Close"],
        probability,
        final_score if "final_score" in locals() else 0,
        signal
    )


st.divider()

show_ai_judgment(
    signal,
    probability
)


   
    # =========================
# チャート表示
# =========================

st.divider()

st.subheader(
    "📈 Project X チャート"
)


if data is not None:

    fig = create_chart(
        data
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )
    
    # =========================
# AIおすすめランキング
# =========================

st.divider()



ranking_codes = [
    "7203",
    "6758",
    "9984",
    "8306",
    "9432",
    "7011",
    "6857",
    "6146",
    "8035",
    "4063",
    "6501",
    "6098",
    "8058",
    "8001",
    "8766",
    "8411",
    "4502",
    "5108",
    "6367",
    "9433"
]


ranking_df = create_ranking(
    ai.model,
    ranking_codes
)

# =========================
# Version 4.7
# AI注目銘柄TOP5
# =========================

st.divider()

if not ranking_df.empty:

    st.subheader(
        "🚀 今日のAI注目銘柄 TOP5"
    )

    top5 = ranking_df.head(5)

    st.dataframe(
        top5,
        width="stretch"
    )
    st.divider()

    st.subheader("👑 今日のNo.1銘柄")

    best = top5.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "銘柄",
        best["コード"]
    )

    c2.metric(
        "AI確率",
        f"{best['AI上昇確率(%)']}%"
    )

    c3.metric(
        "判断",
        best["判断"]
    )

    st.success(
        f"本日のAIイチオシは {best['コード']} です。"
    )

    if "判断" in top5.columns:

        buy_list = top5[
            top5["判断"] == "🔴 BUY"
        ]

        if len(buy_list) > 0:

            st.success(
                "🔥 AI買い候補あり"
            )

            for code in buy_list["コード"]:

                st.write(f"⭐ {code}")

        else:

            st.info("現在強い買い候補なし")

else:

    st.info(
        "ランキング計算できませんでした"
    )

    # =========================
# ポートフォリオ管理
# =========================

st.divider()

st.subheader(
    "📦 My Portfolio"
)


portfolio = [

    {
        "code": "9519",
        "name": "PowerX",
        "amount": 500,
        "buy": 2373
    },

    {
        "code": "7203",
        "name": "トヨタ",
        "amount": 100,
        "buy": 3000
    }

]


portfolio_df, total_profit = calculate_portfolio(
    portfolio
)


if not portfolio_df.empty:

    st.dataframe(
        portfolio_df,
        width="stretch"
    )


    if total_profit >= 0:

        st.success(
            f"🟢 合計損益 +{total_profit:,.0f}円"
        )

    else:

        st.error(
            f"🔴 合計損益 {total_profit:,.0f}円"
        )


else:

    st.info(
        "保有データなし"
    )
    # =========================
# AIバックテスト
# =========================

st.divider()

st.subheader(
    "📊 AIバックテスト"
)


backtest_code = st.text_input(
    "検証銘柄コード",
    "7203",
    key="backtest_code"
)


backtest_period = st.selectbox(
    "検証期間",
    [
        "1y",
        "3y",
        "5y"
    ],
    key="backtest_period"
)


if st.button(
    "📈 バックテスト開始"
):


    result = run_backtest(
        ai.model,
        backtest_code,
        backtest_period
    )


    if not result.empty:


        st.dataframe(
            result,
            width="stretch"
        )


        wins = len(
            result[
                result["勝敗"]=="WIN"
            ]
        )


        total = len(
            result
        )


        win_rate = (
            wins /
            total *
            100
        )


        st.success(
            f"""
検証回数：{total}回

勝率：{win_rate:.1f}%
"""
        )


    else:

        st.info(
            "検証データ不足"
        )
if latest is not None:

    final_score = calculate_score(
        latest,
        probability
    )


    st.metric(
        "Project X スコア",
        f"{final_score} / 100"
    )


    # =========================
# Version 3.3
# AI実績分析パネル
# =========================

show_history()


show_stock_results()


show_simulation()


# =========================
# Version 4.3
# 市場環境スコア
# =========================

st.divider()

st.subheader(
    "🌏 市場環境AI"
)


try:

    market_score, markets = get_market_score()


    st.metric(
        "市場環境スコア",
        f"{market_score}/100"
    )


    for name, value in markets.items():

        st.write(
            f"{name} : {value}"
        )


    if market_score >= 70:

        st.success(
            "🟢 全体相場は追い風"
        )

    elif market_score >= 40:

        st.warning(
            "🟡 慎重な相場"
        )

    else:

        st.error(
            "🔴 リスク高め"
        )


except Exception as e:

    st.warning(
        "市場分析データ取得不可"
    )


# =========================
# Version 4.8
# AIアラートセンター
# =========================

st.divider()

st.subheader("🚨 AIアラート")

if latest is not None:

    alerts = []

    if latest["RSI"] < 30:
        alerts.append("🔥 RSI30以下（売られすぎ）")

    if latest["MACD"] > latest["Signal"]:
        alerts.append("📈 MACDゴールデンクロス")

    if latest["Close"] > latest["MA25"]:
        alerts.append("✅ MA25突破")

    if probability >= 0.70:
        alerts.append("🤖 AI上昇確率70%以上")


    if len(alerts) == 0:

        st.info(
            "現在大きなシグナルはありません"
        )

    else:

        for alert in alerts:

            st.success(alert)

else:

    st.info(
        "分析後に表示されます"
    )