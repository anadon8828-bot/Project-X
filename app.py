import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# 画面設定
# -----------------------------
st.set_page_config(
    page_title="Project X",
    page_icon="📈",
    layout="wide"
)


st.title("🚀 Project X")
st.subheader("株価分析アプリ")


# -----------------------------
# 入力
# -----------------------------
code = st.text_input(
    "銘柄コード（例：7203）",
    "7203"
)


period = st.selectbox(
    "表示期間",
    [
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "3y",
        "5y"
    ],
    index=3
)


# -----------------------------
# 表示ボタン
# -----------------------------
if st.button("株価を分析"):

    ticker = yf.Ticker(code + ".T")

    data = ticker.history(period=period)


    if data.empty:

        st.error("銘柄が見つかりません")

    else:

        # 移動平均
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


        latest = data.iloc[-1]


        # -----------------------------
        # 指標表示
        # -----------------------------

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "現在値",
            f"{latest['Close']:.2f}円"
        )

        col2.metric(
            "高値",
            f"{latest['High']:.2f}円"
        )

        col3.metric(
            "安値",
            f"{latest['Low']:.2f}円"
        )

        col4.metric(
            "出来高",
            f"{int(latest['Volume']):,}"
        )


        st.divider()


        # -----------------------------
        # チャート
        # -----------------------------

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.75,0.25]
        )


        # ローソク足

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


        # 出来高

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="出来高"
            ),
            row=2,
            col=1
        )


        fig.update_layout(

            height=800,

            title=f"{code} 株価チャート",

            xaxis_rangeslider_visible=False

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # 表

        st.subheader("📊 最新データ")

        st.dataframe(
            data.tail(20)
        )