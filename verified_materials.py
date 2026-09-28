"""TDnet issuer disclosures; headline classification, never earnings surprise inference."""
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup

BASE = "https://www.release.tdnet.info/inbs/"


def classify(title):
    if any(x in title for x in ("訂正", "中止", "終了", "取得状況", "取得結果", "経過")):
        return "要確認", False
    if any(x in title for x in ("下方修正", "減配", "無配", "不正", "債務超過")):
        return "注意材料", False
    if "上方修正" in title or "増配" in title:
        return "プラス材料候補", True
    if ("自己株式" in title and "取得" in title and any(x in title for x in ("決定", "設定"))) or "大口受注" in title:
        return "プラス材料候補", True
    return "要確認", False


def parse_page(html, day, allowed_codes, now):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#main-list-table")
    if table is None:
        if 'に開示された情報はありません。' in soup.get_text(' ',strip=True):
            return [], set()
        raise ValueError("TDnet一覧の構造が変わったため取得を停止しました")
    rows = []
    for row in table.select("tr"):
        code_cell, time_cell, title_cell = (row.select_one(x) for x in (".kjCode", ".kjTime", ".kjTitle"))
        if not all((code_cell, time_cell, title_cell)):
            continue
        raw = code_cell.get_text(strip=True)
        code = raw[:-1] if len(raw) == 5 and raw.endswith("0") else raw
        link = title_cell.find("a", href=True)
        if code not in allowed_codes or link is None:
            continue
        url = urljoin(BASE, link["href"])
        if urlparse(url).hostname != "www.release.tdnet.info" or not urlparse(url).path.endswith(".pdf"):
            continue
        published = datetime.fromisoformat(f"{day}T{time_cell.get_text(strip=True)}:00+09:00")
        if not timedelta(0) <= now - published <= timedelta(hours=72):
            continue
        title = link.get_text(strip=True)
        category, positive = classify(title)
        rows.append({"code":code,"title":title,"published":published.isoformat(),"url":url,"source":"TDnet（発行会社の適時開示）","category":category,"positive":positive})
    pages = set(re.findall(r"I_list_\d{3}_\d{8}\.html", html))
    return rows, pages


def collect_materials(allowed_codes, now=None):
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    session = requests.Session()
    def get(name):
        response = session.get(urljoin(BASE,name),timeout=20)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    main = BeautifulSoup(get("I_main_00.html"),"html.parser")
    days = sorted({o.get("value", "") for o in main.select("option") if re.fullmatch(r"I_list_001_\d{8}\.html",o.get("value", ""))},reverse=True)
    if not days:
        raise ValueError("TDnetの公開日を確認できません")
    records = []
    for first in days:
        date = datetime.strptime(first[-13:-5],"%Y%m%d").date()
        if not 0 <= (now.date()-date).days <= 3:
            continue
        html = get(first)
        rows, pages = parse_page(html,str(date),allowed_codes,now)
        records.extend(rows)
        for page in sorted(pages-{first}):
            if page.endswith(first[-13:]):
                records.extend(parse_page(get(page),str(date),allowed_codes,now)[0])
    unique = {r["url"]:r for r in records}
    return sorted(unique.values(),key=lambda r:r["published"],reverse=True)
