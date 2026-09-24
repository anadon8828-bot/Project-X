"""Read-only home views of saved research results."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

def read_saved(name):
    path = ROOT / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"コード": str, "code": str})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"保存データを読み込めません: {name} ({exc})")
        return pd.DataFrame()

def open_ranked_stock(code):
    """Set navigation before the next full application render."""
    st.session_state.active_analysis = {"code": code, "period_name": "1年"}


def render_legacy_ranking():
    st.subheader("本日の注目銘柄 TOP10（全東証対象）")
    history = read_saved("tse_refined_top10.csv")
    candidates = read_saved("tse_all_scan_candidates.csv")
    universe = read_saved("tse_domestic_common_stocks.csv")
    processed = read_saved("tse_all_scan_processed.csv")
    st.caption("東証の内国普通株を一次選抜し、上位候補を詳細評価した研究ランキングです。ETF・REIT等は対象外です。購入を推奨する順位ではありません。")
    if "コード" in universe:
        universe_codes = set(universe["コード"].dropna().astype(str))
        processed_codes = set(processed.get("コード", pd.Series(dtype=str)).dropna().astype(str))
        st.caption(f"対象一覧：{len(universe_codes):,}銘柄 ／ スキャン処理済み：{len(universe_codes & processed_codes):,}銘柄 ／ 一次候補：{len(candidates):,}銘柄")
        st.caption("処理済み件数にはデータ取得不可・条件除外を含みます。全銘柄の評価成功を意味しません。")
        if "コード" in history:
            history = history[history["コード"].isin(universe_codes)].copy()
    if not history.empty:
        saved_at = pd.Timestamp((ROOT / "tse_refined_top10.csv").stat().st_mtime, unit="s", tz="Asia/Tokyo")
        history["分析日時"] = saved_at
    if history.empty or "分析日時" not in history:
        st.info("全東証の詳細評価結果はまだありません。全東証スキャン・詳細評価の完了後に表示します。20銘柄版への置き換えは行いません。")
    else:
        dates = pd.to_datetime(history["分析日時"], errors="coerce")
        latest = dates.max()
        if pd.notna(latest):
            ranking = history.loc[dates == latest].copy()
            if "順位" in ranking:
                ranking = ranking.sort_values("順位")
            ranking = ranking.head(10)
            st.caption(f"結果ファイル更新：{latest:%Y/%m/%d %H:%M}（日本時間）。価格の基準日時とは異なります。")
            if latest.date() != pd.Timestamp.now(tz="Asia/Tokyo").date():
                st.info("本日分は未更新です。直近の保存結果を表示しています。")
            if "コード" in ranking and not ranking.empty:
                companies = read_saved("tse_domestic_common_stocks.csv")
                names = {}
                if {"コード", "銘柄名"}.issubset(companies.columns):
                    names = dict(zip(companies["コード"].str.strip().str.upper(), companies["銘柄名"]))
                st.caption("銘柄を押すと、そのまま詳細を開きます。")
                for position, (_, row) in enumerate(ranking.iterrows(), start=1):
                    if pd.isna(row["コード"]):
                        continue
                    code = str(row["コード"]).strip().upper()
                    name = names.get(code, row.get("銘柄名", "会社名未取得"))
                    if pd.isna(name) or not str(name).strip():
                        name = "会社名未取得"
                    st.button(
                        f"{position}. {code}　{name}",
                        key=f"home_rank_open_{position}_{code}",
                        use_container_width=True,
                        on_click=open_ranked_stock,
                        args=(code,),
                    )
                    details = [str(row[c]) for c in ("市場", "業種", "総合判断", "ニュース") if c in row and pd.notna(row[c]) and str(row[c]).strip()]
                    if details:
                        st.caption(" ／ ".join(details))
        else:
            st.info("ランキングの保存日時を確認できません。")
def render_home_research():
    from candidate_watchlist import render_watchlist
    render_watchlist(read_saved, ROOT, open_ranked_stock)
    with st.expander("翌日予測の記録・成績", expanded=False):
        predictions = read_saved("next_day_prediction_history.csv")
        results = read_saved("next_day_prediction_results.csv")
        st.write(f"予測保存：{len(predictions)}件 ／ 照合済み：{len(results)}件")
        if not predictions.empty and "prediction_date" in predictions:
            st.caption(f"予測基準日の最新：{predictions['prediction_date'].max()}")
        if results.empty:
            st.info("照合済み実績はまだありません。記録の保存と実績照合は別の処理です。")
        else:
            required = {"actual_return_pct", "predicted_return_pct", "actual_date", "code"}
            if required.issubset(results.columns):
                table = results.copy()
                table["誤差（ポイント）"] = (pd.to_numeric(table.actual_return_pct, errors="coerce") - pd.to_numeric(table.predicted_return_pct, errors="coerce")).abs()
                st.caption("保存済み予測と実績の比較です。少数の実績だけでは精度を判断できません。")
                st.dataframe(table[["actual_date", "code", "predicted_return_pct", "actual_return_pct", "誤差（ポイント）"]].rename(columns={"actual_date":"実績日", "code":"コード", "predicted_return_pct":"予測騰落率(%)", "actual_return_pct":"実績騰落率(%)"}), hide_index=True, use_container_width=True)
