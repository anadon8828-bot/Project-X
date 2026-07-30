import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime


# =========================
# 設定
# =========================

st.set_page_config(
    page_title="Project X",
    page_icon="🚀",
    layout="wide"
)


st.title("🚀 Project X")
st.subheader("AI株価分析システム")


# =========================
# モデル読み込み
# =========================

if not os.path.exists("model.pkl"):

    st.error("model.pkl がありません")
    st.stop()


model = joblib.load(
    "model.pkl"
)


# =========================
# 入力
# =========================

code = st.text_input(
    "銘柄コード",
    "7203"
)


period = st.selectbox(
    "期間",
    [
        "6mo",
        "1y",
        "3y",
        "5y"
    ]
)


# =========================
# 分析
# =========================

if st.button("🚀 分析開始"):


    data = yf.Ticker(
        code + ".T"
    ).history(
        period=period
    )


    if data.empty:

        st.error(
            "株価データ取得失敗"
        )

        st.stop()


    # -------------------------
    # テクニカル
    # -------------------------

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


    delta = data["Close"].diff()


    gain = delta.clip(
        lower=0
    )


    loss = -delta.clip(
        upper=0
    )


    rs = (
        gain.rolling(14).mean()
        /
        loss.rolling(14).mean()
    )


    data["RSI"] = (
        100 -
        100/(1+rs)
    )


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
        ema12-ema26
    )


    data["Signal"] = (
        data["MACD"]
        .ewm(span=9)
        .mean()
    )


    latest = data.iloc[-1]


    # -------------------------
    # AI判定
    # -------------------------

    ai_input = pd.DataFrame(
        [[
            latest["MA25"],
            latest["MA75"],
            latest["RSI"],
            latest["MACD"],
            latest["Volume"]
        ]],
        columns=[
            "MA25",
            "MA75",
            "RSI",
            "MACD",
            "Volume"
        ]
    )


    probability = (
        model.predict_proba(
            ai_input
        )[0][1]
    )


    st.divider()

    st.subheader(
        "🤖 AI予測"
    )


    col1,col2 = st.columns(2)


    col1.metric(
        "上昇確率",
        f"{probability*100:.1f}%"
    )


    if probability >= 0.65:

        col2.success(
            "🟢 BUY"
        )

    elif probability >= 0.5:

        col2.warning(
            "🟡 WAIT"
        )

    else:

        col2.error(
            "🔴 SELL"
        )
            # =========================
    # 信頼度表示
    # =========================

    st.divider()

    st.subheader(
        "⭐ AI信頼度"
    )


    if probability >= 0.8:

        st.success(
            "★★★★★ 非常に高い"
        )

    elif probability >= 0.65:

        st.success(
            "★★★★☆ 高い"
        )

    elif probability >= 0.5:

        st.warning(
            "★★★☆☆ 普通"
        )

    elif probability >= 0.4:

        st.info(
            "★★☆☆☆ 低い"
        )

    else:

        st.error(
            "★☆☆☆☆ かなり低い"
        )


    # =========================
    # Project X スコア
    # =========================

    score = 0


    if latest["Close"] > latest["MA25"]:

        score += 1


    if latest["MA25"] > latest["MA75"]:

        score += 2


    if latest["MACD"] > latest["Signal"]:

        score += 2


    if 30 <= latest["RSI"] <= 70:

        score += 1



    st.divider()

    st.subheader(
        "📊 Project X 総合スコア"
    )


    st.metric(
        "スコア",
        f"{score}/6"
    )


    # =========================
    # 基本情報
    # =========================

    st.divider()

    st.subheader(
        "📌 現在データ"
    )


    c1,c2,c3,c4 = st.columns(4)


    c1.metric(
        "現在値",
        f"{latest['Close']:.2f}円"
    )


    c2.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )


    c3.metric(
        "MACD",
        f"{latest['MACD']:.2f}"
    )


    c4.metric(
        "出来高",
        f"{int(latest['Volume']):,}"
    )
        # =========================
    # AI判断理由
    # =========================

    st.divider()

    st.subheader(
        "🔎 AI判断理由"
    )


    if latest["Close"] > latest["MA25"]:

        st.write(
            "✅ 株価はMA25より上"
        )

    else:

        st.write(
            "⚠ 株価はMA25より下"
        )


    if latest["MA25"] > latest["MA75"]:

        st.write(
            "✅ 中期トレンド上向き"
        )

    else:

        st.write(
            "⚠ 中期トレンド弱い"
        )


    if latest["MACD"] > latest["Signal"]:

        st.write(
            "✅ MACD買いシグナル"
        )

    else:

        st.write(
            "⚠ MACD弱い"
        )


    # =========================
    # 履歴保存
    # =========================

    file = "prediction_history.csv"


    record = pd.DataFrame(
        [{
            "date":
                datetime.now().strftime("%Y-%m-%d"),

            "code":
                code,

            "price":
                round(
                    latest["Close"],
                    2
                ),

            "AI確率":
                round(
                    probability*100,
                    1
                ),

            "score":
                score
        }]
    )


    if os.path.exists(file):

        old = pd.read_csv(file)

        record = pd.concat(
            [
                old,
                record
            ],
            ignore_index=True
        )


    record.to_csv(
        file,
        index=False
    )


    # =========================
    # チャート
    # =========================

    st.divider()

    st.subheader(
        "📈 Project X チャート"
    )


    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True
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
        height=900,
        xaxis_rangeslider_visible=False
    )
   
    st.plotly_chart(
        fig,
        width="stretch"
    )


    # =========================
    # 履歴表示
    # =========================

    st.divider()

    st.subheader(
        "📚 AI予測履歴"
    )


    st.dataframe(
        record,
        width="stretch"
    )
    # =========================
# AIおすすめランキング Version 1.1
# =========================

st.divider()

st.subheader(
    "🏆 AIおすすめランキング"
)


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


ranking = []


for c in ranking_codes:

    try:

        df = yf.Ticker(
            c + ".T"
        ).history(
            period="1y"
        )


        if len(df) < 75:

            continue


        df["MA25"] = (
            df["Close"]
            .rolling(25)
            .mean()
        )


        df["MA75"] = (
            df["Close"]
            .rolling(75)
            .mean()
        )


        delta = df["Close"].diff()


        gain = delta.clip(
            lower=0
        )


        loss = -delta.clip(
            upper=0
        )


        rs = (
            gain.rolling(14).mean()
            /
            loss.rolling(14).mean()
        )


        rsi = (
            100 -
            100/(1+rs)
        ).iloc[-1]


        ema12 = (
            df["Close"]
            .ewm(span=12)
            .mean()
        )


        ema26 = (
            df["Close"]
            .ewm(span=26)
            .mean()
        )


        macd = (
            ema12 - ema26
        ).iloc[-1]


        latest_rank = df.iloc[-1]


        input_data = pd.DataFrame(
            [[
                latest_rank["MA25"],
                latest_rank["MA75"],
                rsi,
                macd,
                latest_rank["Volume"]
            ]],
            columns=[
                "MA25",
                "MA75",
                "RSI",
                "MACD",
                "Volume"
            ]
        )


        probability = (
            model.predict_proba(
                input_data
            )[0][1]
        )


        ranking.append(
            {
                "銘柄コード": c,
                "AI上昇確率(%)":
                    round(
                        probability*100,
                        1
                    )
            }
        )


    except:

        pass



if ranking:


    ranking_df = pd.DataFrame(
        ranking
    )


    ranking_df = ranking_df.sort_values(
        "AI上昇確率(%)",
        ascending=False
    )


    st.dataframe(
        ranking_df,
        width="stretch"
    )

else:

    st.info(
        "ランキング計算できませんでした"
    )
    # =========================
# AI成績分析 Version 1.2
# =========================

st.divider()

st.subheader(
    "📈 AI予測成績"
)


history_file = "prediction_history.csv"


if os.path.exists(history_file):

    history = pd.read_csv(
        history_file
    )


    if len(history) > 0:


        total = len(history)


        st.metric(
            "予測回数",
            f"{total}回"
        )


        if "result" in history.columns:


            win = len(
                history[
                    history["result"] == "WIN"
                ]
            )


            lose = len(
                history[
                    history["result"] == "LOSE"
                ]
            )


            count = win + lose


            if count > 0:


                win_rate = (
                    win / count * 100
                )


                c1,c2,c3 = st.columns(3)


                c1.metric(
                    "勝率",
                    f"{win_rate:.1f}%"
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
                    "まだ結果判定前です"
                )


        st.dataframe(
            history,
            width="stretch"
        )


else:

    st.info(
        "まだAI予測履歴がありません"
    )
    # =========================
# Version 1.3
# 急騰・急落検知
# =========================

st.divider()

st.subheader(
    "⚡ 急騰・急落検知"
)


try:

    latest_volume = latest["Volume"]

    avg_volume = (
        data["Volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )


    volume_ratio = (
        latest_volume /
        avg_volume
        *
        100
    )


    price_change = (
        data["Close"].pct_change()
        .iloc[-1]
        *
        100
    )


    # -------------------------
    # 急騰判定
    # -------------------------

    if (
        price_change >= 3
        and
        volume_ratio >= 150
        and
        latest["MACD"] > latest["Signal"]
    ):

        st.success(
            f"""
🚀 急騰シグナル

前日比：
{price_change:.2f}%

出来高：
通常比 {volume_ratio:.0f}%

MACD：
上昇転換

短期買い優勢
"""
        )


    # -------------------------
    # 急落判定
    # -------------------------

    elif (
        price_change <= -3
        and
        volume_ratio >= 150
    ):

        st.error(
            f"""
🔻 急落警戒

前日比：
{price_change:.2f}%

出来高：
通常比 {volume_ratio:.0f}%

売り圧力増加
"""
        )


    else:

        st.info(
            f"""
⚪ 通常状態

前日比：
{price_change:.2f}%

出来高：
通常比 {volume_ratio:.0f}%
"""
        )


except:

    st.warning(
        "急騰判定データ不足"
    )
    # =========================
# Version 1.4
# 短期売買モード
# =========================

st.divider()

st.subheader(
    "🎯 短期売買モード"
)


try:

    current_price = latest["Close"]


    target_price = (
        current_price * 1.05
    )


    stop_price = (
        current_price * 0.97
    )


    trade_score = 0


    if probability >= 0.65:

        trade_score += 2


    if latest["MACD"] > latest["Signal"]:

        trade_score += 2


    if latest["Close"] > latest["MA25"]:

        trade_score += 1


    if volume_ratio >= 120:

        trade_score += 1



    if trade_score >= 5:

        st.success(
            f"""
🟢 短期エントリー候補

総合点：
{trade_score}/6

現在値：
{current_price:.2f}円

利確目安：
{target_price:.2f}円

損切り目安：
{stop_price:.2f}円
"""
        )


    elif trade_score >= 3:

        st.warning(
            f"""
🟡 監視銘柄

総合点：
{trade_score}/6

現在値：
{current_price:.2f}円

様子見推奨
"""
        )


    else:

        st.error(
            f"""
🔴 見送り

総合点：
{trade_score}/6

短期材料不足
"""
        )


except:

    st.warning(
        "短期判定データ不足"
    )
    # =========================
# Version 1.5
# 保有銘柄管理
# =========================

st.divider()

st.subheader(
    "📦 保有銘柄管理"
)


hold_code = st.text_input(
    "保有銘柄コード",
    "9519"
)


hold_amount = st.number_input(
    "保有株数",
    min_value=0,
    value=100,
    step=100
)


buy_price = st.number_input(
    "平均取得価格",
    min_value=0.0,
    value=1000.0,
    step=10.0
)


if st.button(
    "📊 保有分析"
):

    try:

        hold_data = yf.Ticker(
            hold_code + ".T"
        ).history(
            period="5d"
        )


        now_price = (
            hold_data["Close"]
            .iloc[-1]
        )


        profit = (
            now_price - buy_price
        ) * hold_amount


        profit_rate = (
            (now_price-buy_price)
            /
            buy_price
            *
            100
        )


        c1,c2,c3 = st.columns(3)


        c1.metric(
            "現在値",
            f"{now_price:.2f}円"
        )


        c2.metric(
            "評価損益",
            f"{profit:,.0f}円"
        )


        c3.metric(
            "損益率",
            f"{profit_rate:.2f}%"
        )


        st.divider()


        if probability >= 0.65:


            st.success(
                """
🟢 AI判断：保有継続

上昇期待あり
"""
            )


        elif probability >= 0.5:


            st.warning(
                """
🟡 AI判断：様子見

慎重判断
"""
            )


        else:


            st.error(
                """
🔴 AI判断：見直し検討

下落リスクあり
"""
            )


    except:

        st.error(
            "分析できませんでした"
        )
        # =========================
# Version 1.6
# ポートフォリオ管理
# =========================

st.divider()

st.subheader(
    "📊 My Portfolio"
)


portfolio = [
    {
        "code":"9519",
        "name":"PowerX",
        "amount":500,
        "buy":2373
    },
    {
        "code":"7203",
        "name":"トヨタ",
        "amount":100,
        "buy":3000
    }
]


portfolio_data = []


total_profit = 0


for stock in portfolio:

    try:

        price_data = yf.Ticker(
            stock["code"] + ".T"
        ).history(
            period="5d"
        )


        now = (
            price_data["Close"]
            .iloc[-1]
        )


        profit = (
            now - stock["buy"]
        ) * stock["amount"]


        profit_rate = (
            (now-stock["buy"])
            /
            stock["buy"]
            *
            100
        )


        total_profit += profit


        portfolio_data.append(
            {
                "銘柄":
                    stock["name"],

                "コード":
                    stock["code"],

                "株数":
                    stock["amount"],

                "現在値":
                    round(now,2),

                "損益":
                    round(profit),

                "損益率":
                    round(
                        profit_rate,
                        2
                    )
            }
        )


    except:

        pass



if portfolio_data:


    portfolio_df = pd.DataFrame(
        portfolio_data
    )


    st.dataframe(
        portfolio_df,
        width="stretch"
    )


    st.divider()


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
        "保有銘柄データなし"
    )
    # =========================
# Version 1.7
# AI銘柄スクリーニング
# =========================

st.divider()

st.subheader(
    "🔍 今日のAI注目銘柄"
)


screen_result = []


for c in ranking_codes:

    try:

        df = yf.Ticker(
            c + ".T"
        ).history(
            period="1y"
        )


        if len(df) < 75:
            continue


        df["MA25"] = (
            df["Close"]
            .rolling(25)
            .mean()
        )


        df["MA75"] = (
            df["Close"]
            .rolling(75)
            .mean()
        )


        latest_s = df.iloc[-1]


        score_s = 0


        if latest_s["Close"] > latest_s["MA25"]:

            score_s += 1


        if latest_s["MA25"] > latest_s["MA75"]:

            score_s += 2


        input_s = pd.DataFrame(
            [[
                latest_s["MA25"],
                latest_s["MA75"],
                50,
                0,
                latest_s["Volume"]
            ]],
            columns=[
                "MA25",
                "MA75",
                "RSI",
                "MACD",
                "Volume"
            ]
        )


        ai_s = (
            model.predict_proba(
                input_s
            )[0][1]
        )


        screen_result.append(
            {
                "コード": c,

                "AI確率(%)":
                    round(
                        ai_s*100,
                        1
                    ),

                "スコア":
                    score_s
            }
        )


    except:

        pass



if screen_result:


    screen_df = pd.DataFrame(
        screen_result
    )


    screen_df = screen_df.sort_values(
        [
            "AI確率(%)",
            "スコア"
        ],
        ascending=False
    )


    st.dataframe(
        screen_df,
        width="stretch"
    )


else:

    st.info(
        "スクリーニング結果なし"
    )    
    # =========================
# Version 1.8
# AI材料チェック（簡易版）
# =========================

st.divider()

st.subheader(
    "📰 AI材料チェック"
)


news_code = st.text_input(
    "材料確認する銘柄コード",
    "9519",
    key="news_check_code"
)


if st.button(
    "📰 材料分析する"
):

    try:

        ticker_news = yf.Ticker(
            news_code + ".T"
        )


        news = ticker_news.news


        if news:

            positive_words = [
                "上方修正",
                "増益",
                "提携",
                "契約",
                "受注",
                "成長",
                "黒字"
            ]


            negative_words = [
                "赤字",
                "下方修正",
                "減益",
                "不祥事",
                "損失"
            ]


            positive = 0
            negative = 0


            for item in news:

                title = str(
                    item.get(
                        "title",
                        ""
                    )
                )


                for word in positive_words:

                    if word in title:

                        positive += 1


                for word in negative_words:

                    if word in title:

                        negative += 1



            st.write(
                "取得ニュース数：",
                len(news)
            )


            if positive > negative:

                st.success(
                    f"""
🟢 ポジティブ材料

好材料：
{positive}件

悪材料：
{negative}件

AI判断：材料面は強め
"""
                )


            elif negative > positive:

                st.error(
                    f"""
🔴 ネガティブ材料

好材料：
{positive}件

悪材料：
{negative}件

AI判断：注意
"""
                )


            else:

                st.info(
                    f"""
⚪ 材料中立

好材料：
{positive}件

悪材料：
{negative}件
"""
                )


            st.write(
                "最新ニュース"
            )


            for item in news[:5]:

                st.write(
                    "・",
                    item.get(
                        "title",
                        ""
                    )
                )


        else:

            st.info(
                "ニュース取得できませんでした"
            )


    except Exception as e:

        st.warning(
            "材料分析エラー"
        )
        # =========================
# Version 1.9
# AI総合スコア
# =========================

st.divider()

st.subheader(
    "🧠 Project X 最終判断"
)


try:

    final_score = 0


    # AI評価
    ai_point = (
        ai_probability * 40
    )


    final_score += ai_point



    # トレンド評価

    if latest["Close"] > latest["MA25"]:

        final_score += 15


    if latest["MA25"] > latest["MA75"]:

        final_score += 15



    # MACD評価

    if latest["MACD"] > latest["Signal"]:

        final_score += 15



    # RSI評価

    if 40 <= latest["RSI"] <= 70:

        final_score += 10



    # 出来高評価

    if volume_ratio >= 120:

        final_score += 5



    final_score = int(
        min(
            final_score,
            100
        )
    )


    st.metric(
        "Project X 総合スコア",
        f"{final_score} / 100"
    )


    if final_score >= 80:

        st.success(
            """
🚀 強気判定

AI・テクニカルともに良好
"""
        )


    elif final_score >= 60:

        st.info(
            """
🟡 中立〜やや強気

監視候補
"""
        )


    elif final_score >= 40:

        st.warning(
            """
⚪ 様子見

材料不足
"""
        )


    else:

        st.error(
            """
🔴 弱気判定

リスク注意
"""
        )


except:

    st.warning(
        "総合スコア計算不可"
    )
    # =========================
# Version 2.0
# AI売買判断パネル
# =========================

st.divider()

st.subheader(
    "🚦 Project X 最終判断"
)


try:

    if final_score >= 80:

        decision = "🟢 BUY"

        message = """
強気シグナル

AI・テクニカルとも良好
"""

    elif final_score >= 60:

        decision = "🟡 HOLD"

        message = """
監視継続

上昇余地あり
"""

    elif final_score >= 40:

        decision = "⚪ WAIT"

        message = """
様子見

材料確認が必要
"""

    else:

        decision = "🔴 SELL"

        message = """
弱気シグナル

リスク管理優先
"""


    col_a, col_b = st.columns(2)


    col_a.metric(
        "AI判断",
        decision
    )


    col_b.metric(
        "信頼度",
        f"{final_score}%"
    )


    st.info(
        message
    )


except:

    st.warning(
        "判断データ不足"
    )
    # =========================
# Version 2.1
# お気に入り監視リスト
# =========================

st.divider()

st.subheader(
    "⭐ My Watch List"
)


watch_codes = st.text_input(
    "監視銘柄コード（カンマ区切り）",
    "9519,2160,6857",
    key="watch_list"
)


if st.button(
    "⭐ 監視分析する"
):


    watch_result = []


    codes = [
        x.strip()
        for x in watch_codes.split(",")
    ]


    for c in codes:


        try:


            watch_data = yf.Ticker(
                c + ".T"
            ).history(
                period="6mo"
            )


            if len(watch_data) < 75:

                continue


            close = (
                watch_data["Close"]
                .iloc[-1]
            )


            ma25 = (
                watch_data["Close"]
                .rolling(25)
                .mean()
                .iloc[-1]
            )


            if close > ma25:

                status = "🟢 上昇傾向"

            else:

                status = "🔴 弱い"



            watch_result.append(
                {
                    "コード": c,
                    "現在値":
                        round(close,2),
                    "判断":
                        status
                }
            )


        except:

            pass



    if watch_result:


        watch_df = pd.DataFrame(
            watch_result
        )


        st.dataframe(
            watch_df,
            width="stretch"
        )


    else:

        st.info(
            "分析できる銘柄がありません"
        )
        # =========================
# Version 2.2
# バックテスト
# =========================

st.divider()

st.subheader(
    "📊 AIバックテスト"
)


backtest_code = st.text_input(
    "バックテスト銘柄",
    "7203",
    key="backtest_code"
)


backtest_period = st.selectbox(
    "期間",
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

    st.info(
        "バックテストを準備中..."
    )

    try:

        test_data = yf.Ticker(
            backtest_code + ".T"
        ).history(
            period=backtest_period
        )

        st.success(
            f"データ取得成功：{len(test_data)}日"
        )

        st.dataframe(
            test_data.tail(10),
            width="stretch"
        )

    except Exception as e:

        st.error(
            "バックテスト失敗"
        )