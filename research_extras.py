"""Explicit risk arithmetic and provider-reported event dates; no orders."""
import math
import pandas as pd
import streamlit as st
import yfinance as yf


def risk_numbers(price,stop,budget,capital,cost_pct):
    if not all(math.isfinite(x) for x in (price,stop,budget,capital,cost_pct)) or not 0<stop<price or budget<=0 or capital<=0 or cost_pct<0:
        raise ValueError('購入価格より低い正の損切り価格と、正の予算・運用資金を入力してください。')
    shares=int(budget//(price*100))*100
    invested=shares*price
    loss=shares*(price-stop)+invested*cost_pct/100
    return shares,invested,loss,loss/capital*100


def render_risk_check(code,price,capital,holdings,root):
    with st.expander('売買前のリスク確認（現物買い・発注しません）'):
        budget=st.number_input('この銘柄への購入予算（円）',min_value=0.0,value=100000.0,step=10000.0,key='risk_budget_'+code)
        stop=st.number_input('自分で設定する損切り価格（円）',min_value=0.0,value=0.0,step=1.0,key='risk_stop_'+code)
        costs=st.number_input('往復コストの仮定（%）',min_value=0.0,value=0.15,step=0.05,key='risk_cost_'+code)
        st.caption(f'試算の購入価格：{price:,.2f}円（日足基準・リアルタイムではありません）。損切り価格は自動推奨しません。')
        if stop>0:
            try:
                shares,invested,loss,ratio=risk_numbers(price,stop,budget,capital,costs)
                st.write(f'100株単位の試算：{shares:,}株 ／ 購入額 {invested:,.0f}円 ／ 想定損失 {loss:,.0f}円（運用資金の{ratio:.2f}%）')
                if shares==0:
                    st.warning('この予算では100株を購入できません。単元未満株はこの試算の対象外です。')
                if invested>capital:
                    st.warning('購入額が運用資金を超えています。')
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.info('損切り価格を入力すると損失額を計算します。')
        st.warning('窓開け・急変・流動性不足で損切り価格より不利な約定になる場合があり、損失上限は保証されません。信用取引には使えません。')
        try:
            universe=pd.read_csv(root/'tse_domestic_common_stocks.csv',dtype={'コード':str}).set_index('コード')
            sector=universe.loc[code,'業種']
            same=holdings[holdings['コード'].astype(str).map(universe['業種']).eq(sector)]
            unknown=holdings['コード'].astype(str).map(universe['業種']).isna().sum()
            exposure=(pd.to_numeric(same['株数'])*pd.to_numeric(same['取得単価'])).sum()
            st.caption(f'同業種「{sector}」の登録保有：{len(same)}ロット、取得金額合計 {exposure:,.0f}円（時価ではなく、信用売りも相殺しない合計）。業種不明 {unknown}ロット。')
        except (OSError,KeyError,ValueError):
            st.caption('業種偏り：取得未確認')


@st.cache_data(ttl=900,show_spinner=False)
def event_dates(ticker):
    calendar=yf.Ticker(ticker).calendar
    if not isinstance(calendar,dict):
        raise ValueError('予定日の配信形式を確認できません')
    result=[]
    for field,label in [('Earnings Date','決算発表予定（配信元予想を含む）'),('Ex-Dividend Date','配当落ち日'),('Dividend Date','配当支払日')]:
        values=calendar.get(field)
        if values is None:
            result.append({'イベント':label,'日付':'取得未確認'})
            continue
        for value in values if isinstance(values,(list,tuple)) else [values]:
            date=pd.to_datetime(value,errors='coerce')
            result.append({'イベント':label,'日付':str(date.date()) if pd.notna(date) else '取得未確認'})
    return result,pd.Timestamp.now(tz='Asia/Tokyo').isoformat()


def render_events(ticker):
    with st.expander('決算・配当イベント（予定の確認）'):
        try:
            rows,checked=event_dates(ticker)
            st.dataframe(pd.DataFrame(rows),hide_index=True)
            st.caption(f'取得日時：{checked} ／ 配信元：Yahoo Finance。予定は変更される場合があります。会社のIR原文で最終確認してください。')
            today=pd.Timestamp.now(tz='Asia/Tokyo').date()
            for row in rows:
                date=pd.to_datetime(row['日付'],errors='coerce')
                if pd.notna(date) and 0<=(date.date()-today).days<=7:
                    st.warning(f"7日以内：{row['イベント']} {row['日付']}。発表前後の価格急変に注意。")
        except Exception as exc:
            st.info(f'イベント予定は取得未確認です。「予定なし」とは判断しません。（{exc}）')
