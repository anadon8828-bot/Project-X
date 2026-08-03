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
from history import save_prediction
from analysis import calculate_analysis
from charts import create_chart
from signals import calculate_rebound_score, create_signal, create_decision
from dashboard import show_ai_judgment


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

decision = create_decision(
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
        # =========================
# Project X 総合AIスコア
# =========================

st.divider()

st.subheader(
    "🧠 Project X 総合判断"
)


try:

    final_score = 0


    # AI評価（最大40点）

    final_score += (
        probability * 40
    )


    # トレンド（最大30点）

    if latest["Close"] > latest["MA25"]:

        final_score += 15


    if latest["MA25"] > latest["MA75"]:

        final_score += 15



    # MACD（15点）

    if latest["MACD"] > latest["Signal"]:

        final_score += 15



    # RSI（10点）

    if 40 <= latest["RSI"] <= 70:

        final_score += 10



    final_score = int(
        min(
            final_score,
            100
        )
    )


    st.metric(
        "Project X スコア",
        f"{final_score} / 100"
    )



    if final_score >= 80:

        st.success(
            """
🚀 強気判定（上昇）

AI・テクニカルとも良好
"""
        )


    elif final_score >= 60:

        st.info(
            """
🟡 監視候補

上昇余地あり
"""
        )


    elif final_score >= 40:

        st.warning(
            """
⚪ 様子見

材料確認
"""
        )


    else:

        st.error(
            """
🟢 弱気判定（下落警戒）

リスク注意
"""
        )


except:

    st.warning(
        "スコア計算不可"
    )
    
   
# =========================
# Version 3.1
# Project X ダッシュボード
# =========================

st.divider()

st.subheader(
    "🚀 Project X ダッシュボード"
)


try:

    dash1, dash2, dash3, dash4 = st.columns(4)


    dash1.metric(
        "現在値",
        f"{latest['Close']:.2f}円"
    )


    dash2.metric(
        "AI上昇確率",
        f"{probability*100:.1f}%"
    )


    dash3.metric(
        "総合スコア",
        f"{final_score}/100"
    )


    if final_score >= 80:

        dash4.error(
            "🔴 BUY"
        )

    elif final_score >= 60:

        dash4.warning(
            "🟡 HOLD"
        )

    else:

        dash4.success(
            "🟢 WAIT"
        )


except:

    st.info(
        "分析後に表示されます"
    )

# =========================
# Version 5.1
# AI診断カード
# =========================

st.divider()

st.subheader(
    "🧠 AI診断"
)


try:

    if final_score >= 80:

        grade = "★★★★★ 強気"

    elif final_score >= 60:

        grade = "★★★★ 期待"

    elif final_score >= 40:

        grade = "★★★ 中立"

    else:

        grade = "★★ 注意"


    st.success(
        grade
    )


    diagnosis = []


    if probability >= 0.65:

        diagnosis.append(
            "✅ AI上昇確率が高い"
        )

    else:

        diagnosis.append(
            "⚠ AI上昇確率は低め"
        )


    if latest["MACD"] > latest["Signal"]:

        diagnosis.append(
            "✅ MACD買いシグナル"
        )

    else:

        diagnosis.append(
            "⚠ MACD弱い"
        )


    if latest["Close"] > latest["MA25"]:

        diagnosis.append(
            "✅ 短期トレンド上向き"
        )

    else:

        diagnosis.append(
            "⚠ 株価は移動平均以下"
        )


    if latest["RSI"] < 30:

        diagnosis.append(
            "🔥 売られすぎ反発期待"
        )

    elif latest["RSI"] > 70:

        diagnosis.append(
            "⚠ 過熱注意"
        )


    for item in diagnosis:

        st.write(item)


except:

    st.info(
        "分析後に表示されます"
    )

try:

    if final_score >= 80:

        color_message = """
🔴 強気ゾーン

買い検討レベル
"""

    elif final_score >= 60:

        color_message = """
🟡 監視ゾーン

タイミング待ち
"""

    else:

        color_message = """
🟢 慎重ゾーン

リスク管理優先
"""


    st.info(
        color_message
    )

except:

    st.info(
        "分析後に表示されます"
    )
    
    # =========================
# Version 3.3
# AI実績分析パネル
# =========================

st.divider()

st.subheader(
    "🏆 AI実績分析"
)


try:

    history_file = "prediction_history.csv"


    if os.path.exists(history_file):

        history_df = pd.read_csv(
            history_file
        )


        total = len(history_df)


        st.metric(
            "AI予測回数",
            f"{total}回"
        )


        if "result" in history_df.columns:


            win = len(
                history_df[
                    history_df["result"] == "WIN"
                ]
            )


            lose = len(
                history_df[
                    history_df["result"] == "LOSE"
                ]
            )


            finished = win + lose


            if finished > 0:


                rate = (
                    win /
                    finished *
                    100
                )


                c1,c2,c3 = st.columns(3)


                c1.metric(
                    "勝率",
                    f"{rate:.1f}%"
                )


                c2.metric(
                    "勝ち",
                    f"{win}回"
                )


                c3.metric(
                    "負け",
                    f"{lose}回"
                )


            else:

                st.info(
                    "まだ結果判定待ちです"
                )


        st.dataframe(
            history_df,
            width="stretch"
        )


    else:

        st.info(
            "まだAI予測履歴がありません"
        )


except:

    st.warning(
        "実績分析エラー"
    )
    # =========================
# Version 3.4
# AI銘柄別成績分析
# =========================

st.divider()

st.subheader(
    "🧠 AI銘柄別成績"
)


try:

    history_file = "prediction_history.csv"


    if os.path.exists(history_file):

        df_result = pd.read_csv(
            history_file
        )


        if "code" in df_result.columns and "result" in df_result.columns:


            result_df = df_result[
                df_result["result"].isin(
                    [
                        "WIN",
                        "LOSE"
                    ]
                )
            ]


            if len(result_df) > 0:


                stock_result = []


                for stock_code in result_df["code"].unique():


                    stock_data = result_df[
                        result_df["code"] == stock_code
                    ]


                    wins = len(
                        stock_data[
                            stock_data["result"] == "WIN"
                        ]
                    )


                    total = len(
                        stock_data
                    )


                    rate = (
                        wins /
                        total *
                        100
                    )


                    stock_result.append(
                        {
                            "銘柄コード":
                                stock_code,

                            "予測回数":
                                total,

                            "勝率(%)":
                                round(
                                    rate,
                                    1
                                )
                        }
                    )


                stock_df = pd.DataFrame(
                    stock_result
                )


                stock_df = stock_df.sort_values(
                    "勝率(%)",
                    ascending=False
                )


                st.dataframe(
                    stock_df,
                    width="stretch"
                )


            else:

                st.info(
                    "判定済みデータがありません"
                )


        else:

            st.info(
                "分析データ不足"
            )


    else:

        st.info(
            "履歴なし"
        )


except:

    st.warning(
        "銘柄別分析エラー"
    )
    # =========================
# Version 3.6
# 仮想100万円運用シミュレーション
# =========================

st.divider()

st.subheader(
    "💰 Project X 仮想運用"
)


try:

    start_money = 1000000


    if os.path.exists(
        "prediction_history.csv"
    ):


        sim = pd.read_csv(
            "prediction_history.csv"
        )


        if "result" in sim.columns:


            win_count = len(
                sim[
                    sim["result"]=="WIN"
                ]
            )


            lose_count = len(
                sim[
                    sim["result"]=="LOSE"
                ]
            )


            total = (
                win_count +
                lose_count
            )


            if total > 0:


                win_rate = (
                    win_count /
                    total
                )


                # 1回あたり仮想利益5%
                # 負け3%

                estimated = (
                    start_money *
                    (
                        1
                        +
                        (
                            win_rate*0.05
                            -
                            (1-win_rate)*0.03
                        )
                        *
                        total
                    )
                )


                profit = (
                    estimated -
                    start_money
                )


                c1,c2,c3 = st.columns(3)


                c1.metric(
                    "開始資金",
                    "100万円"
                )


                c2.metric(
                    "評価額",
                    f"{estimated:,.0f}円"
                )


                c3.metric(
                    "損益",
                    f"{profit:+,.0f}円"
                )


            else:

                st.info(
                    "まだ検証データ不足"
                )


        else:

            st.info(
                "結果判定待ち"
            )


except:

    st.warning(
        "シミュレーション計算不可"
    )
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