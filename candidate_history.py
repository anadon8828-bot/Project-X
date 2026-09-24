"""Immutable candidate observations with next-session-open forward outcomes."""
import sqlite3
import json
import hashlib
import pandas as pd
from research_rules import tokyo_calendar


def connect(root):
    db=sqlite3.connect(root/'candidate_observations.sqlite',timeout=20)
    db.execute('CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, recorded TEXT NOT NULL, payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS outcomes (id TEXT, code TEXT, payload TEXT, PRIMARY KEY(id,code))')
    return db


def record(root,rows):
    if rows.empty:
        return
    payload=rows.to_json(orient='records',force_ascii=False)
    key=hashlib.sha256(payload.encode()).hexdigest()
    db=connect(root)
    try:
        with db:
            db.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?)',(key,pd.Timestamp.now(tz='Asia/Tokyo').isoformat(),payload))
    finally:
        db.close()


def evaluate_prices(recorded,prices,now):
    cal=tokyo_calendar()
    start=pd.Timestamp(recorded).tz_convert('Asia/Tokyo')
    day=cal.date_to_session(str(start.date()),direction='next')
    if cal.session_open(day)<=start.tz_convert('UTC'):
        day=cal.next_session(day)
    sessions=cal.sessions_window(day,5)
    # sessions_window includes start and returns five sessions in the installed API.
    sessions=sessions[:5]
    bars=prices.copy()
    bars.index=pd.to_datetime(bars.index).tz_localize(None).normalize()
    completed=[s for s in sessions if cal.session_close(s)+pd.Timedelta(minutes=30)<=now.tz_convert('UTC')]
    result={'状態':'保留','基準':'記録後の最初の寄付から測定（約定保証なし・コスト控除前）'}
    if not completed or day not in bars.index:
        return result
    entry=float(bars.loc[day,'Open'])
    if not entry>0:
        return {'状態':'価格取得不可'}
    result.update({'開始日':str(day.date()),'開始価格':entry,'翌営業日騰落率(%)':(float(bars.loc[day,'Close'])/entry-1)*100})
    if len(completed)==5 and all(s in bars.index for s in sessions):
        window=bars.loc[sessions]
        if window[['Open','High','Low','Close']].isna().any().any():
            return {'状態':'価格取得不可'}
        result.update({'状態':'確定','5営業日騰落率(%)':(float(window.Close.iloc[-1])/entry-1)*100,'最大上昇(%)':(float(window.High.max())/entry-1)*100,'最大下落(%)':(float(window.Low.min())/entry-1)*100})
    return result


def settle(root):
    import yfinance as yf
    db=connect(root)
    try:
        snapshots=db.execute('SELECT id,recorded,payload FROM snapshots ORDER BY recorded').fetchall()
        cached={}
        now=pd.Timestamp.now(tz='Asia/Tokyo')
        for key,recorded,payload in snapshots:
            for row in json.loads(payload):
                code=str(row['コード'])
                old=db.execute('SELECT payload FROM outcomes WHERE id=? AND code=?',(key,code)).fetchone()
                if old and json.loads(old[0]).get('状態')=='確定':
                    continue
                if pd.Timestamp(recorded).date()>=now.date():
                    continue
                try:
                    if code not in cached:
                        data=yf.download(code+'.T',start=pd.Timestamp(recorded).strftime('%Y-%m-%d'),auto_adjust=True,progress=False)
                        if isinstance(data.columns,pd.MultiIndex):
                            data.columns=data.columns.get_level_values(0)
                        cached[code]=data
                    result=evaluate_prices(recorded,cached[code],now)
                except Exception as exc:
                    result={'状態':'取得未確認','詳細':str(exc)}
                with db:
                    db.execute('INSERT OR REPLACE INTO outcomes VALUES (?,?,?)',(key,code,json.dumps(result,ensure_ascii=False)))
    finally:
        db.close()


def render(root):
    import streamlit as st
    db=connect(root)
    try:
        snapshots=db.execute('SELECT recorded,payload FROM snapshots ORDER BY recorded DESC LIMIT 2').fetchall()
        outcomes=db.execute('SELECT s.recorded,o.code,o.payload,s.payload FROM outcomes o JOIN snapshots s ON o.id=s.id ORDER BY s.recorded DESC').fetchall()
        count=db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0]
    finally:
        db.close()
    with st.expander('候補の変更理由・掲載後実績'):
        st.caption(f'候補記録：{count}回。掲載時の価格・理由・取得日時を固定保存。自動更新完了時と画面表示時に記録します。PC・更新処理が停止中は蓄積しません。')
        if len(snapshots)==2:
            current={str(r['コード']):r for r in json.loads(snapshots[0][1])}
            previous={str(r['コード']):r for r in json.loads(snapshots[1][1])}
            st.write('新規候補：'+('、'.join(sorted(current.keys()-previous.keys())) or 'なし'))
            st.write('候補外：'+('、'.join(sorted(previous.keys()-current.keys())) or 'なし'))
            changed=[c for c in current.keys()&previous.keys() if current[c].get('候補理由')!=previous[c].get('候補理由')]
            st.write('候補理由の変更：'+('、'.join(sorted(changed)) or 'なし'))
            st.caption('候補外は他銘柄との条件比較・取得状況でも変わります。悪材料の発生と断定しません。')
        done=[]
        for recorded,code,payload,saved in outcomes:
            original=next((r for r in json.loads(saved) if str(r['コード'])==code),{})
            done.append({'記録日時':recorded,'コード':code,'分類':'材料あり' if original.get('材料候補') else 'テクニカルのみ','候補理由':original.get('候補理由',''),**json.loads(payload)})
        if done:
            table=pd.DataFrame(done)
            st.dataframe(table,hide_index=True)
            completed=table[table['状態']=='確定']
            if not completed.empty:
                st.caption('候補分類別の観測集計（重複期間を含む・コスト控除前・優位性の証明ではありません）')
                st.dataframe(completed.groupby('分類')['5営業日騰落率(%)'].agg(['count','mean']).rename(columns={'count':'観測件数','mean':'平均騰落率(%)'}))
        else:
            st.info('確定実績はまだありません。記録後の最初の寄付から測定し、将来価格が揃うまで保留します。旧ランキングの過去実績とは混ぜません。')
        st.caption('同じ銘柄の重複期間を含む観測です。独立した売買回数・資金曲線・勝率として扱いません。')
