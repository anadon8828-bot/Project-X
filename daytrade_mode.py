"""Next-session day-trade watchlist, not a price prediction or trade signal."""
import json
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import exchange_calendars as xcals
import yfinance as yf
from vwap_chart import with_vwap

US_SYMBOLS='AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO NFLX PLTR COIN MSTR INTC MU QCOM ARM ORCL CRM UBER JPM BAC GS XOM CVX WMT COST DIS PYPL SOFI'.split()


def dates(market,now=None):
    now=now or pd.Timestamp.now(tz='UTC')
    cal=xcals.get_calendar('XTKS' if market=='JP' else 'XNYS')
    local=now.tz_convert('Asia/Tokyo' if market=='JP' else 'America/New_York')
    session=cal.date_to_session(str(local.date()),direction='previous')
    target=cal.next_session(session)
    completed=session
    if cal.session_close(session)+pd.Timedelta(minutes=30)>now:
        completed=cal.previous_session(session)
    return str(completed.date()),str(target.date())


def evaluate_frame(frame,code,name,market,expected):
    f=frame.dropna(subset=['Open','High','Low','Close','Volume']).sort_index()
    f=f[[str(d.date())<=expected for d in f.index]]
    if len(f)<21 or str(f.index[-1].date())!=expected:
        return None
    tail=f.tail(20)
    price=float(f.Close.iloc[-1])
    turnover=float((tail.Close*tail.Volume).mean())
    spread=float(((tail.High-tail.Low)/tail.Close*100).mean())
    base=float(f.Volume.iloc[-21:-1].mean())
    relative=float(f.Volume.iloc[-1]/base) if base>0 else np.nan
    if not all(np.isfinite(v) for v in [price,turnover,spread,relative]) or price<=0 or turnover<(100_000_000 if market=='JP' else 20_000_000) or not 1<=spread<=12:
        return None
    return {'コード':code,'会社名':name,'終値':price,'平均売買代金':turnover,'平均日中値幅(%)':spread,'出来高倍率':relative,'価格日':expected,'前日比(%)':float(f.Close.pct_change().iloc[-1]*100)}


def rank(rows):
    if not rows:
        return []
    df=pd.DataFrame(rows)
    # Heuristic research priority, not probability; reduce reward for extreme ranges.
    df['priority']=df['平均売買代金'].rank(pct=True)*.4+df['出来高倍率'].clip(0,3).rank(pct=True)*.4+df['平均日中値幅(%)'].clip(0,5).rank(pct=True)*.2
    return df.sort_values(['priority','コード'],ascending=[False,True]).head(5).drop(columns='priority').to_dict('records')


def scan(market,root):
    expected,target=dates(market)
    if market=='JP':
        from tse_universe import load_universe
        u=load_universe(refresh=True)
        names=dict(zip(u['コード'].astype(str),u['銘柄名']))
    else:
        names={s:s for s in US_SYMBOLS}
    rows=[]
    failed=0
    symbols=list(names)
    for offset in range(0,len(symbols),60):
        batch=symbols[offset:offset+60]
        tickers=[s+'.T' if market=='JP' else s for s in batch]
        raw=yf.download(tickers,period='3mo',group_by='ticker',auto_adjust=True,threads=8,progress=False,timeout=15)
        for code,ticker in zip(batch,tickers):
            try:
                frame=raw[ticker] if isinstance(raw.columns,pd.MultiIndex) else raw
                result=evaluate_frame(frame,code,names[code],market,expected)
                if result:
                    rows.append(result)
            except (KeyError,ValueError,TypeError,IndexError):
                failed+=1
        print(f'{market}: {min(offset+60,len(symbols))}/{len(symbols)}',flush=True)
    selected=rank(rows)
    if market=='US':
        for row in selected:
            try:
                info=yf.Ticker(row['コード']).get_info()
                row['会社名']=info.get('longName') or row['コード']
            except Exception:
                pass
    payload={'created':pd.Timestamp.now(tz='UTC').isoformat(),'price_day':expected,'target_day':target,'universe':len(symbols),'eligible':len(rows),'errors':failed,'rows':selected}
    path=root/f'daytrade_{market}.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def render_daytrade(market,root):
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    st.subheader(('日本株' if market=='JP' else '米国株')+'・翌営業日のデイトレ監視候補5選')
    expected,target=dates(market)
    st.caption(f'対象営業日：{target} ／ 使用する確定日足：{expected}。取引中の場合、前営業日までの暫定選定です。市場現地日付で判定します。')
    st.warning('未検証の監視リストです。翌日の利益・値動きは予測しません。遅延配信のため、発注前のリアルタイム価格・板・スプレッドは証券会社で確認してください。')
    st.caption('対象：東証内国普通株一覧' if market=='JP' else '対象：米国の大型・活発な30銘柄の固定リスト（米国全銘柄ではありません）：'+', '.join(US_SYMBOLS))
    st.caption('条件：20日平均売買代金が日本株1億円／米国株2千万ドル以上、平均日中値幅1〜12%。優先度は売買代金40%・出来高倍率40%・値幅20%の相対順位。上昇確率や期待利益ではありません。')
    if st.button('最新データで候補5選を作成',key='daytrade_scan_'+market):
        with st.spinner('対象銘柄を取得・評価しています。日本株は数分かかる場合があります…'):
            try:
                scan(market,root)
            except Exception as exc:
                st.error(f'候補更新に失敗：{exc}')
                return
    try:
        payload=json.loads((root/f'daytrade_{market}.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        st.info('候補は未作成です。上のボタンで作成してください。')
        return
    age=pd.Timestamp.now(tz='UTC')-pd.Timestamp(payload['created'])
    if payload['price_day']!=expected or payload['target_day']!=target or not pd.Timedelta(0)<=age<=pd.Timedelta(minutes=30):
        st.warning('保存済み候補の有効期限が切れています。更新するまで表示しません。')
        return
    st.caption(f"取得：{pd.Timestamp(payload['created']).tz_convert('Asia/Tokyo')} ／ 対象 {payload['universe']}・条件適合 {payload['eligible']}・処理例外 {payload['errors']}。適合以外には欠損・古い日付・条件除外を含みます。")
    rows=payload['rows']
    if not rows:
        st.info('条件を満たす銘柄はありません。無理に5銘柄に埋めません。')
        return
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    names={r['コード']:r['会社名'] for r in rows}
    symbol=st.selectbox('チャートを見る候補',list(names),format_func=lambda c:c+'　'+names[c],key='daytrade_code_'+market)
    st.caption('候補理由は上表の流動性・日中値幅・出来高倍率です。決算・ニュース・寄付ギャップは選定未反映で、翌朝の再確認が必要です。')
    interval=st.radio('時間足',['1分足','5分足','15分足'],index=1,horizontal=True,key='daytrade_interval_'+market)
    chart_style=st.radio('チャート表示',['終値の折れ線','ローソク足'],horizontal=True,key='daytrade_chart_style_'+market)
    vwap_style=st.radio('VWAP表示',['表示期間を連続','1日ごとにリセット'],horizontal=True,key='daytrade_vwap_style_'+market)
    try:
        ticker=symbol+'.T' if market=='JP' else symbol
        frame=yf.download(ticker,period='5d',interval={'1分足':'1m','5分足':'5m','15分足':'15m'}[interval],auto_adjust=True,prepost=False,progress=False)
        if isinstance(frame.columns,pd.MultiIndex):
            frame.columns=frame.columns.get_level_values(0)
        frame=frame.dropna(subset=['Open','High','Low','Close','Volume'])
        if frame.empty:
            raise ValueError('分足データなし')
        market_tz='Asia/Tokyo' if market=='JP' else 'America/New_York'
        daily_vwap=vwap_style=='1日ごとにリセット'
        frame=with_vwap(frame,daily_vwap,market_tz)
        frame=frame.tail(400)
        local_index=pd.DatetimeIndex(frame.index)
        if local_index.tz is None:
            local_index=local_index.tz_localize(market_tz)
        else:
            local_index=local_index.tz_convert(market_tz)
        session_day=pd.Series(local_index.date,index=frame.index)
        vwap_plot=frame.VWAP.mask(session_day.ne(session_day.shift())) if daily_vwap else frame.VWAP
        # 非取引時間を詰め、スマホでも値動きを大きく表示する。
        chart_x=local_index.strftime('%m/%d %H:%M')
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.75,.25])
        if chart_style=='終値の折れ線':
            fig.add_trace(go.Scatter(x=chart_x,y=frame.Close,name='終値',mode='lines',line=dict(color='#4cc9f0',width=2)),row=1,col=1)
        else:
            fig.add_trace(go.Candlestick(x=chart_x,open=frame.Open,high=frame.High,low=frame.Low,close=frame.Close,name='株価'),row=1,col=1)
        vwap_name=('日別VWAP（近似）' if daily_vwap else '表示期間VWAP（近似）')
        fig.add_trace(go.Scatter(x=chart_x,y=vwap_plot,name=vwap_name,mode='lines',connectgaps=False,line=dict(color='#ff7fbf',width=2)),row=1,col=1)
        fig.add_trace(go.Bar(x=chart_x,y=frame.Volume,name='出来高',marker_color='#526078'),row=2,col=1)
        fig.update_layout(template='plotly_dark',height=560,xaxis_rangeslider_visible=False,margin=dict(l=8,r=8,t=20,b=8))
        fig.update_xaxes(type='category',nticks=8)
        fig.update_yaxes(title_text='円' if market=='JP' else '米ドル',row=1,col=1)
        st.plotly_chart(fig,use_container_width=True)
        basis=('市場現地日付で毎日リセット' if daily_vwap else '表示期間の先頭から連続計算')
        st.caption(f'最終足：{frame.index[-1]}。通常取引時間の分足近似VWAP・{basis}。約定データから計算する証券会社のVWAPとは一致しない場合があります。')
    except Exception as exc:
        st.warning(f'チャート取得不可：{exc}')


if __name__=='__main__':
    scan(sys.argv[1],Path(__file__).resolve().parent)
