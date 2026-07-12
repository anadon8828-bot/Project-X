import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Project X",
    page_icon="📈",
    layout="wide"
)


st.title("🚀 Project X")
st.subheader("株価分析アプリ")


code = st.text_input(
    "銘柄コード（例：7203）",
    "7203"
)


period = st.selectbox(
    "表示期間",
    ["3mo", "6mo", "1y", "3y", "5y"],
    index=2
)


if st.button("株価を分析"):

    ticker = yf.Ticker(code + ".T")

    data = ticker.history(period=period)


    if data.empty:

        st.error("銘柄が見つかりません")

    else:

        # 移動平均
        data["MA25"] = data["Close"].rolling(25).mean()
        data["MA75"] = data["Close"].rolling(75).mean()


        # RSI計算
        delta = data["Close"].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        data["RSI"] = 100 - (100 / (1 + rs))


        # MACD計算
        ema12 = data["Close"].ewm(span=12).mean()
        ema26 = data["Close"].ewm(span=26).mean()

        data["MACD"] = ema12 - ema26
        data["Signal"] = data["MACD"].ewm(span=9).mean()


        latest = data.iloc[-1]


        # 指標表示

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "現在値",
            f"{latest['Close']:.2f}円"
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


        # 判定

        st.divider()

        st.subheader("🤖 Project X 判定")


        if latest["RSI"] > 70:

            st.warning(
                "RSI：買われすぎ注意"
            )

        elif latest["RSI"] < 30:

            st.success(
                "RSI：売られすぎの可能性"
            )

        else:

            st.info(
                "RSI：適正範囲"
            )


        if latest["MACD"] > latest["Signal"]:

            st.success(
                "MACD：上昇トレンド傾向"
            )

        else:

            st.warning(
                "MACD：下落トレンド傾向"
            )


        # チャート

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5,0.25,0.25]
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


        # RSI

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                name="RSI"
            ),
            row=2,
            col=1
        )


        # MACD

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


        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,
            title=f"{code} Project X分析"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )