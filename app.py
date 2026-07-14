import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================
# 設定
# =============================

st.set_page_config(
    page_title="Project X",
    page_icon="🚀",
    layout="wide"
)


# =============================
# タイトル
# =============================

st.title("🚀 Project X")
st.subheader("AI株価分析アプリ")


# =============================
# 入力
# =============================

code = st.text_input(
    "銘柄コード",
    "7203"
)

period = st.selectbox(
    "分析期間",
    ["3mo", "6mo", "1y", "3y", "5y"],
    index=2
)


# =============================
# 分析開始
# =============================

if st.button("🚀 分析する"):

    ticker = yf.Ticker(code + ".T")

    data = ticker.history(
        period=period
    )


    if data.empty:

        st.error(
            "銘柄が見つかりません"
        )


    else:

        # =============================
        # テクニカル計算
        # =============================

        data["MA25"] = (
            data["Close"]
            .rolling(25)
            .mean()
        )


        data["MA75"] = (
            data["Close"]
            .rolling(75)
            .mean()
        )


        # RSI

        delta = data["Close"].diff()

        gain = delta.where(
            delta > 0,
            0
        )

        loss = -delta.where(
            delta < 0,
            0
        )


        avg_gain = (
            gain
            .rolling(14)
            .mean()
        )

        avg_loss = (
            loss
            .rolling(14)
            .mean()
        )


        rs = avg_gain / avg_loss


        data["RSI"] = (
            100 -
            (100 / (1 + rs))
        )


        # MACD

        ema12 = (
            data["Close"]
            .ewm(span=12)
            .mean()
        )

        ema26 = (
            data["Close"]
            .ewm(span=26)
            .mean()
        )


        data["MACD"] = (
            ema12 - ema26
        )


        data["Signal"] = (
            data["MACD"]
            .ewm(span=9)
            .mean()
        )


        latest = data.iloc[-1]


        # 前日比

        previous = data.iloc[-2]

        change = (
            latest["Close"]
            -
            previous["Close"]
        )


        change_percent = (
            change
            /
            previous["Close"]
            *
            100
        )
                # =============================
        # 基本情報表示
        # =============================

        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "現在値",
            f"{latest['Close']:.2f}円",
            f"{change:+.2f}円 ({change_percent:+.2f}%)"
        )


        col2.metric(
            "RSI",
            f"{latest['RSI']:.2f}"
        )


        col3.metric(
            "MACD",
            f"{latest['MACD']:.2f}"
        )


        col4.metric(
            "出来高",
            f"{int(latest['Volume']):,}"
        )


        # =============================
        # Project X スコア
        # =============================

        score = 0


        if latest["Close"] > latest["MA25"]:
            score += 1


        if latest["MA25"] > latest["MA75"]:
            score += 2


        if 30 <= latest["RSI"] <= 70:
            score += 1


        if latest["MACD"] > latest["Signal"]:
            score += 2



        # =============================
        # AI判定
        # =============================

        st.divider()

        st.subheader(
            "🤖 Project X AI分析"
        )


        st.metric(
            "総合スコア",
            f"{score} / 6"
        )


        if score >= 5:

            st.success(
                """
🟢 買い優勢

★★★★★

・移動平均良好
・MACD上昇傾向
・RSI正常範囲

短期：強気
中期：強気
"""
            )


        elif score >= 3:

            st.info(
                """
🟡 中立

★★★

・上昇と下落材料が混在

慎重判断
"""
            )


        else:

            st.error(
                """
🔴 売り優勢

★★

・弱いシグナルが多い

様子見
"""
            )


        # =============================
        # チャート作成
        # =============================

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


        # 株価

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


        # MA25

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA25"],
                name="MA25"
            ),
            row=1,
            col=1
        )


        # MA75

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA75"],
                name="MA75"
            ),
            row=1,
            col=1
        )
                # =============================
        # RSI
        # =============================

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                name="RSI"
            ),
            row=2,
            col=1
        )


        # =============================
        # MACD
        # =============================

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD"],
                name="MACD"
            ),
            row=3,
            col=1
        )


        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Signal"],
                name="Signal"
            ),
            row=3,
            col=1
        )


        # =============================
        # チャート設定
        # =============================

        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,
            title=f"{code} Project X分析"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )