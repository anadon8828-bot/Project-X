"""Evidence-based watchlist selection, independent of AI buy thresholds."""
import pandas as pd
import streamlit as st
import json
from research_rules import fresh_rows, valid_materials, assessment, expected_price_day


def render_candidate_detail(root, code):
    st.subheader("候補理由・売買条件（ホームと共通）")
    try:
        prices = fresh_rows(pd.read_csv(root / 'watchlist_fresh_candidates.csv',dtype={'コード':str}))
        snapshot = json.loads((root / 'verified_materials.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        st.warning('共通の最新判定を取得できません。売買条件は判定不可です。')
        return
    matches = prices[prices['コード']==code] if 'コード' in prices else pd.DataFrame()
    if matches.empty:
        st.warning('最新の全東証候補データにこの銘柄はありません。取得不可・条件除外・期限切れを含みます。売買条件は判定不可です。')
        return
    news = [m for m in valid_materials(snapshot) if m['code']==code]
    result = assessment(matches.iloc[0],news)
    st.write(result['現在の判断'])
    st.write('候補理由：'+result['候補理由'])
    st.caption('注意点：'+result['注意点'])
    st.caption(f"共通判定の価格日：{matches.iloc[0]['株価基準日']} ／ 取得日時：{matches.iloc[0]['取得日時']}")
    for item in news:
        st.link_button(f"{item['title']}（{item['published']}／TDnet原文）",item['url'])


def select_watchlist(candidates, limit=10, materials=None):
    if candidates.empty or "コード" not in candidates:
        return pd.DataFrame()
    rows = candidates.dropna(subset=["コード"]).drop_duplicates("コード").copy()
    counts = pd.to_numeric(rows.get("テクニカル一致数", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    materials = materials or []
    positive_codes = {m['code'] for m in materials if m.get('positive') is True}
    rows = rows.loc[counts.ge(1) | rows['コード'].isin(positive_codes)].copy()
    rows["一致数"] = counts.loc[rows.index]
    rows["候補理由"] = rows["一致数"].map(lambda n: f"価格・移動平均・MACD・出来高のプラス条件が4項目中{int(n)}項目一致")
    rows["現在の判断"] = "買い検討候補（売買条件は未確認）"
    rows["注意点"] = "最新価格・材料・過熱感・損切り位置の再確認が必要。AIは選定条件に使用していません。"
    rows["売買代金"] = pd.to_numeric(rows.get("平均売買代金(百万円)", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    for index,row in rows.iterrows():
        for key,value in assessment(row,[m for m in materials if m['code']==str(row['コード'])]).items():
            rows.loc[index,key] = value
    technical = rows.sort_values(["一致数", "売買代金", "コード"], ascending=[False,False,True],na_position="last")
    news = technical[technical['コード'].isin(positive_codes)]
    # Reserve up to half for sourced materials; fill remaining slots by technicals.
    return pd.concat([news.head(limit//2),technical]).drop_duplicates('コード').head(limit)


@st.fragment(run_every="60s")
def render_watchlist(read_saved, root, open_stock):
    import streamlit as st
    st.subheader("注目銘柄TOP10・買い検討候補（全東証対象）")
    candidates = read_saved("watchlist_fresh_candidates.csv")
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    candidates = fresh_rows(candidates, now)
    if candidates.empty:
        st.error("最新性を確認できる候補がありません。古い銘柄・数字は表示しません。更新中・更新失敗・休場の場合も買い判定は行いません。")
        import json
        path = root / "watchlist_update_status.json"
        state = {}
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                st.caption(f"更新状態：{ {'RUNNING':'取得中','FAILED':'更新失敗','COMPLETED':'取得終了'}.get(state.get('state'),'未確認')} ／ 処理済み：{state.get('processed',0)} / {state.get('total','確認中')}")
            except (OSError, ValueError):
                st.caption("更新状況を確認できません。")
        started = pd.to_datetime(state.get("started"), utc=True, errors="coerce")
        retry_allowed = pd.isna(started) or now.tz_convert("UTC") - started > pd.Timedelta(minutes=15)
        heartbeat = pd.to_datetime(state.get('heartbeat',state.get('started')),utc=True,errors='coerce')
        stalled = pd.isna(heartbeat) or now.tz_convert('UTC')-heartbeat>pd.Timedelta(minutes=10)
        if (state.get("state") != "RUNNING" or stalled) and retry_allowed:
            import subprocess
            import sys
            with (root / "watchlist_refresh.log").open("a", encoding="utf-8") as log:
                subprocess.Popen([sys.executable, str(root / "refresh_watchlist.py")], cwd=root, stdout=log, stderr=log, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            st.info("最新データの自動取得を開始しました。完了後にページを更新してください。")
        return
    universe = read_saved("tse_domestic_common_stocks.csv")
    if "コード" in universe and "コード" in candidates:
        candidates = candidates[candidates["コード"].isin(universe["コード"])].copy()
    st.caption(f"対象一覧：{len(universe):,}銘柄 ／ 保存済み一次候補：{len(candidates):,}銘柄。ETF・REIT、取得不可・流動性不足・履歴不足は対象外です。")
    import json
    try:
        state = json.loads((root / "watchlist_update_status.json").read_text(encoding="utf-8"))
        st.caption(f"今回の取得：処理 {state.get('processed',0):,}銘柄 ／ 取得不可・古い足・履歴不足 {state.get('failed',0):,}銘柄 ／ 流動性条件除外 {state.get('excluded',0):,}銘柄")
    except (OSError, ValueError):
        st.warning("取得状況の集計を確認できません。全銘柄の取得成功を保証するものではありません。")
    st.caption("AI確率では足切りしません。公式開示の材料候補を最大5枠、残りはプラス条件数・売買代金順で選びます。利益の期待順位ではありません。")
    st.caption(f"価格基準日：{expected_price_day(now)}（取引所カレンダー）。休場日・寄付前は直近営業日を使用。画面表示中は60秒ごとに確認し、取得から30分超のデータは非表示・再取得します。")
    path = root / "watchlist_fresh_candidates.csv"
    if path.exists():
        stamp = pd.Timestamp(path.stat().st_mtime, unit="s", tz="Asia/Tokyo")
        st.caption(f"結果保存：{stamp:%Y/%m/%d %H:%M}。最新営業日・取得後30分以内のみ表示。遅延配信の日足であり、リアルタイム価格ではありません。")
        if stamp.date() != pd.Timestamp.now(tz="Asia/Tokyo").date():
            st.warning("本日分は未更新です。以下は保存時点の候補であり、現在の買い条件成立を示しません。")
    st.info("買い条件成立：未判定。候補に入ることと、今買うことは別です。開示見出しの分類は業績への影響や市場予想を上回るサプライズの確認ではありません。")
    try:
        snapshot = json.loads((root / 'verified_materials.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        snapshot = {}
    materials = valid_materials(snapshot,now)
    if snapshot.get('state') != 'COMPLETED' or not materials:
        st.warning('公式開示の確認可能な材料がありません。取得失敗・期限切れの場合も材料なしと断定せず、ニュース加点は行いません。')
    else:
        st.caption(f"公式開示確認：{snapshot.get('checked')} ／ 直近72時間の銘柄一致開示 {len(materials)}件。一般ニュースは未統合です。")
    rows = select_watchlist(candidates,materials=materials)
    from candidate_history import record, render
    try:
        record(root,rows)
        render(root)
    except Exception as exc:
        st.error(f'候補履歴の保存・表示に失敗しました。記録済みとは扱いません：{exc}')
    names = dict(zip(universe.get("コード", []), universe.get("銘柄名", [])))
    if rows.empty:
        st.info("保存データからプラス条件のある候補を抽出できませんでした。市場全体に候補がないという意味ではありません。")
    for position, (_, row) in enumerate(rows.iterrows(), 1):
        code = str(row["コード"])
        if st.button(f"{code}　{names.get(code, row.get('銘柄名', '会社名未取得'))}", key=f"home_rank_open_{position}_{code}", use_container_width=True):
            open_stock(code)
            st.rerun(scope="app")
        st.write(f"候補理由：{row['候補理由']}")
        for item in [m for m in materials if m['code']==code]:
            st.link_button(f"{item['category']}：{item['title']}（{item['published']}／TDnet原文）",item['url'])
        st.caption(f"取得日時：{row.get('取得日時', '不明')}")
        st.caption(f"現在の判断：{row['現在の判断']} ／ 注意点：{row['注意点']}")
        if "株価基準日" in row and pd.notna(row["株価基準日"]):
            st.caption(f"株価基準日：{row['株価基準日']}")
