"""US research view; deliberately isolated from Japanese prediction models."""
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=900, show_spinner=False)
def company_info(symbol):
    return yf.Ticker(symbol).get_info()


def render_us_equities(load_data, load_chart_data, sakata, elliott, chart_patterns):
    st.subheader("米国株リサーチ")
    st.caption("米ドル建て。配信には遅延があり、リアルタイム株価・板情報ではありません。")
    with st.form("us_search"):
        entered = st.text_input("ティッカー", value="AAPL", placeholder="例：AAPL / MSFT / NVDA / BRK-B")
        submitted = st.form_submit_button("米国株を分析", type="primary")
    if submitted:
        symbol = entered.strip().upper().replace(".", "-")
        if not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", symbol):
            st.error("米国株のティッカーを入力してください（例：AAPL、BRK-B）。")
            return
        st.session_state.us_active_symbol = symbol
    symbol = st.session_state.get("us_active_symbol")
    if not symbol:
        st.info("ティッカーを入力すると、会社情報とチャートを表示します。日本株の保有情報とは分けて管理します。")
        return
    try:
        with st.spinner(f"{symbol}の企業情報と日足を取得しています…"):
            info = company_info(symbol)
            if info.get("currency") != "USD" or info.get("quoteType") != "EQUITY":
                st.warning("米ドル建ての株式として確認できません。対象銘柄または配信状況をご確認ください。")
                return
            data = load_data(symbol, "2y")
            if data.empty:
                raise ValueError("分析に必要な株価履歴が不足しています。")
    except Exception as exc:
        st.error(f"米国株データを取得できませんでした：{exc}")
        return
    latest = data.iloc[-1]
    st.subheader(f"{symbol}　{info.get('longName') or info.get('shortName') or '会社名未取得'}")
    st.caption(f"日足の基準日時：{latest.name} ／ 取引所時間帯：{info.get('exchangeTimezoneName', '取得不可')}。当日足は取引中に変化します。")
    a, b, c = st.columns(3)
    a.metric("直近日足価格（USD・調整済み）", f"${latest.Close:,.2f}")
    b.metric("日足騰落率", f"{latest.Return_1D * 100:+.2f}%")
    c.metric("出来高（株）", f"{latest.Volume:,.0f}")
    st.warning("米国株AIは未学習・未検証です。上昇確率・翌日予測・総合スコア・売買判断は表示しません。日本株モデルは使用していません。")
    tabs = st.tabs(["チャート", "テクニカル・波動", "企業情報", "開発状況"])
    with tabs[0]:
        timeframe = st.radio("時間足", ["1分足", "5分足", "15分足", "日足", "週足", "月足"], index=3, horizontal=True, key="us_timeframe")
        try:
            bars = load_chart_data(symbol, timeframe).tail(300)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.76, .24], vertical_spacing=.04)
            fig.add_trace(go.Candlestick(x=bars.index, open=bars.Open, high=bars.High, low=bars.Low, close=bars.Close, name="株価（USD）"), row=1, col=1)
            for name in ("MA25", "MA75"):
                fig.add_trace(go.Scatter(x=bars.index, y=bars[name], name=name), row=1, col=1)
            fig.add_trace(go.Bar(x=bars.index, y=bars.Volume, name="出来高"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=480, margin=dict(l=4,r=4,t=14,b=4), xaxis_rangeslider_visible=False, legend_orientation="h")
            fig.update_yaxes(title_text="米ドル", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":False,"displaylogo":False})
            st.caption(f"時間足の最終日時：{bars.index[-1]}。分足の取得期間は配信元の制約があります。")
        except Exception as exc:
            st.warning(f"チャートを取得できませんでした：{exc}")
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"指標":k,"値":float(latest[k])} for k in ("MA25","MA75","MA200","RSI","MACD","ATR","ADX","VWAP")]), hide_index=True)
        st.caption("価格系指標は米ドル建て。以下は形状の機械的検出で、確率や検証済みの売買シグナルではありません。")
        st.write("酒田五法")
        patterns = sakata(data)
        if patterns:
            st.dataframe(pd.DataFrame(patterns).drop(columns="score", errors="ignore"), hide_index=True)
        else:
            st.info("酒田五法の候補なし")
        wave = elliott(data)
        st.write(f"エリオット波動：{wave['label']}")
        patterns = chart_patterns(data)
        if patterns:
            st.dataframe(pd.DataFrame(patterns), hide_index=True)
        else:
            st.info("チャートパターンの候補なし")
    with tabs[2]:
        st.caption("会社名・業種の配信原文は英語です。財務数値は配信元の最新公表値で、株価とは基準日が異なります。")
        fields = {"取引所":"fullExchangeName","業種（原文）":"industry","PER":"trailingPE","PBR":"priceToBook","EPS（報告通貨）":"trailingEps","財務報告通貨":"financialCurrency","時価総額（USD）":"marketCap","売上高（報告通貨）":"totalRevenue"}
        st.dataframe(pd.DataFrame([{"項目":label,"値":str(info.get(key) if info.get(key) is not None else "取得不可")} for label,key in fields.items()]), hide_index=True)
    with tabs[3]:
        st.info("対応済み：検索・会社名・6種類の時間足・出来高・テクニカル・波動候補・企業情報。")
        st.info("未対応：米国株専用AI・ランキング・ニュース統合・ドル円換算・米国株保有管理・売買検証。日本株の記録やモデルは変更しません。")
