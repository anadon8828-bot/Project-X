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


    # =========================
    # 反発期待度AI
    # =========================

    rebound_score = 0


    if latest["RSI"] < 30:

        rebound_score += 40


    if latest["Close"] < latest["MA25"] * 0.95:

        rebound_score += 30


    if latest["MACD"] > latest["Signal"]:

        rebound_score += 20


    if latest["Volume"] > data["Volume"].rolling(20).mean().iloc[-1]:

        rebound_score += 10



    st.divider()

    st.subheader(
        "📈 反発期待度AI"
    )


    st.progress(
        rebound_score / 100
    )


    st.metric(
        "反発期待度",
        f"{rebound_score}%"
    )


    if rebound_score >= 70:

        st.success(
            "⭐ 強い反発候補"
        )


    elif rebound_score >= 50:

        st.warning(
            "👀 監視候補"
        )


    else:

        st.info(
            "様子見"
        )


    # AI判定

    probability, signal = ai.predict(
        latest
    )
    save_prediction(
    code,
    latest["Close"],
    probability,
    final_score if "final_score" in locals() else 0,
    signal
)


    st.divider()

    st.success(
    f"""
## 🚀 今日のAI判断

### {signal}

AI上昇確率：**{probability*100:.1f}%**
"""
)

    st.subheader(
        "🤖 AI予測"
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "上昇確率",
        f"{probability*100:.1f}%"
    )


    if probability >= 0.65:

        col2.error(
            "🔴 BUY"
        )


    elif probability >= 0.5:

        col2.warning(
            "🟡 WAIT"
        )


    else:

        col2.success(
            "🟢 SELL"
        )

     




        # 現在データ

    st.divider()

    st.subheader(
        "📌 現在データ"
    )


    a,b,c,d = st.columns(4)


    a.metric(
        "現在値",
        f"{latest['Close']:.2f}円"
    )


    b.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )


    c.metric(
        "MACD",
        f"{latest['MACD']:.2f}"
    )


    d.metric(
        "出来高",
        f"{int(latest['Volume']):,}"
    )
    # =========================
    # チャート表示
    # =========================

    st.divider()

    st.subheader(
        "📈 Project X チャート"
    )


    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[
            0.5,
            0.25,
            0.25
        ]
    )


    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="株価"
        ),
        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA25"],
            name="MA25"
        ),
        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA75"],
            name="MA75"
        ),
        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI"
        ),
        row=2,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD"
        ),
        row=3,
        col=1
    )
        


    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False
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

st.subheader(
    "🚀 今日のAI注目銘柄 TOP5"
)


try:

    top5 = ranking_df.head(5)


    if not top5.empty:


        st.dataframe(
            top5,
            width="stretch"
        )


        buy_list = top5[
            top5["判断"] == "🔴 BUY"
        ]


        if len(buy_list) > 0:

            st.success(
                "🔥 AI買い候補あり"
            )

            for code in buy_list["コード"]:

                st.write(
                    f"⭐ {code}"
                )


        else:

            st.info(
                "現在強い買い候補なし"
            )


except:

    st.info(
        "ランキング計算後に表示されます"
    )

if not ranking_df.empty:


    st.divider()

    st.subheader(
        "🚀 今日のAI注目銘柄 TOP5"
    )


    top5 = ranking_df.head(5)


    st.dataframe(
        top5,
        width="stretch"
    )


    buy_list = top5[
        top5["判断"] == "🔴 BUY"
    ]


    if len(buy_list) > 0:

        st.success(
            "🔥 AI買い候補あり"
        )


        for code in buy_list["コード"]:

            st.write(
                f"⭐ {code}"
            )


    else:

        st.info(
            "現在強い買い候補なし"
        )


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
# 最終売買判断パネル
# =========================

st.divider()

st.subheader(
    "🚦 Project X 最終判断"
)


try:


    if final_score >= 80:

        decision = "🔴 BUY"

        comment = """
強気シグナル

AI・テクニカルともに良好
"""


    elif final_score >= 60:

        decision = "🟡 HOLD"

        comment = """
監視継続

上昇余地あり
"""


    elif final_score >= 40:

        decision = "⚪ WAIT"

        comment = """
様子見

材料確認が必要
"""


    else:

        decision = "🟢 SELL"

        comment = """
弱気シグナル

リスク管理優先
"""



    col1,col2 = st.columns(2)


    col1.metric(
        "判断",
        decision
    )

    col2.metric(
        "信頼度",
        f"{final_score}%"
    )

    st.info(
        comment
    )

    st.divider()

    st.subheader("🎯 AI買いポイント")

    buy_price = latest["MA25"]

    take_profit = latest["Close"] * 1.08

    stop_loss = latest["Close"] * 0.95

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "買い目安",
        f"{buy_price:.2f}円"
    )

    c2.metric(
        "利確目安",
        f"{take_profit:.2f}円"
    )

    c3.metric(
        "損切り目安",
        f"{stop_loss:.2f}円"
    )

except:

    st.warning(
        "判断データ不足"
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
# Version 3.2
# 投資判断カード
# =========================

st.divider()

st.subheader(
    "📌 Investment Panel"
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


    st.write(
        "AI分析ポイント"
    )


    points = []


    if probability >= 0.65:

        points.append(
            "✅ AI上昇確率が高い"
        )

    else:

        points.append(
            "⚠ AI確率は低め"
        )


    if latest["Close"] > latest["MA25"]:

        points.append(
            "✅ 短期トレンド上向き"
        )

    else:

        points.append(
            "⚠ 短期トレンド弱い"
        )


    if latest["MACD"] > latest["Signal"]:

        points.append(
            "✅ MACD買い方向"
        )

    else:

        points.append(
            "⚠ MACD弱い"
        )


    for p in points:

        st.write(p)


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
# Version 4.3
# 市場環境AI
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
            "🟢 相場環境 良好"
        )

    elif market_score >= 40:

        st.warning(
            "🟡 慎重な相場"
        )

    else:

        st.error(
            "🔴 リスク高め"
        )


except:

    st.info(
        "市場データ取得待ち"
    )
    # =========================
# Version 4.4
# AIエントリーポイント判定
# =========================

st.divider()

st.subheader(
    "🎯 AIエントリーポイント"
)


try:

    entry_score = 0

    reasons = []


    # RSI判定

    if latest["RSI"] < 35:

        entry_score += 30

        reasons.append(
            "✅ RSI売られすぎ"
        )

    elif latest["RSI"] < 50:

        entry_score += 15

        reasons.append(
            "🟡 RSI改善余地"
        )



    # MACD判定

    if latest["MACD"] > latest["Signal"]:

        entry_score += 30

        reasons.append(
            "✅ MACD反転"
        )


    # 出来高判定

    avg_volume = data["Volume"].rolling(20).mean().iloc[-1]


    if latest["Volume"] > avg_volume:

        entry_score += 20

        reasons.append(
            "✅ 出来高増加"
        )


    # 株価位置

    if latest["Close"] < latest["MA25"]:

        entry_score += 20

        reasons.append(
            "✅ 押し目位置"
        )


    entry_score = min(
        entry_score,
        100
    )


    st.metric(
        "AI買い推奨度",
        f"{entry_score}%"
    )


    st.progress(
        entry_score / 100
    )


    if entry_score >= 70:

        st.success(
            "🚀 買い検討ゾーン"
        )

    elif entry_score >= 40:

        st.warning(
            "👀 監視ゾーン"
        )

    else:

        st.info(
            "⏸ 待機ゾーン"
        )


    st.write(
        "判定理由"
    )


    for r in reasons:

        st.write(r)



    st.write(
        f"""
### 参考価格

現在値：
{latest['Close']:.2f}円

買い目安：
{latest['MA25']:.2f}円

目標：
{latest['Close']*1.08:.2f}円

損切：
{latest['Close']*0.95:.2f}円
"""
    )


except:

    st.info(
        "分析後に表示されます"
    )