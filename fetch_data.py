import os
import re
import time
import hashlib
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_HTML_DIR = os.path.join(BASE_DIR, "data", "raw_html")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(RAW_HTML_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

YEARS = list(range(2015, 2027))  # 2015 to 2025 for training/eval, 2026 for bonus prediction
AWARD_YEARS = list(range(2015, 2026))

TEAM_CODES = {
    "t": ("阪神", "CE"),
    "g": ("巨人", "CE"),
    "yb": ("DeNA", "CE"),
    "c": ("広島", "CE"),
    "s": ("ヤクルト", "CE"),
    "d": ("中日", "CE"),
    "h": ("ソフトバンク", "PA"),
    "l": ("西武", "PA"),
    "f": ("日本ハム", "PA"),
    "bs": ("オリックス", "PA"),
    "m": ("ロッテ", "PA"),
    "e": ("楽天", "PA"),
}

NPB_TEAM_MAP = {
    "神": "阪神",
    "巨": "巨人",
    "ディ": "DeNA",
    "デ": "DeNA",
    "広": "広島",
    "ヤ": "ヤクルト",
    "中": "中日",
    "ソ": "ソフトバンク",
    "西": "西武",
    "日": "日本ハム",
    "オ": "オリックス",
    "ロ": "ロッテ",
    "楽": "楽天",
}

TEAM_TO_LEAGUE = {v[0]: v[1] for v in TEAM_CODES.values()}

KANJI_REPL = str.maketrans({
    "﨑": "崎",
    "髙": "高",
    "海": "海",
    "濱": "浜",
    "濵": "浜",
    "邉": "辺",
    "邊": "辺",
    "齋": "斎",
    "齊": "斉",
    "國": "国",
    "嶋": "島",
    "嶌": "島",
    "澤": "沢",
    "塚": "塚",
    "\ufa11": "崎",
})


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name)
    s = s.translate(KANJI_REPL)
    # Remove spaces
    s = re.sub(r"[\s　]+", "", s)
    # Remove leading initials like "L.", "J.A.", "R.", "T."
    s = re.sub(r"^[A-Za-z.・]+", "", s)
    return s.strip()


def year_prefix(year: int) -> str:
    if year == 2026:
        return ""
    return f"{str(year)[2:]}/"


def fetch_url(url: str, retries: int = 3) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(RAW_HTML_DIR, f"{h}.html")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 500:
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                text = resp.text
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return text
            elif resp.status_code == 404:
                return ""
        except Exception as e:
            if attempt == retries - 1:
                print(f"[WARN] Failed {url}: {e}")
            time.sleep(0.5 * (attempt + 1))
    return ""


def collect_all_urls():
    urls = []
    # 1. NPB Best Nine pages (2015-2025)
    for y in AWARD_YEARS:
        urls.append(f"https://npb.jp/award/{y}/voting_bt9.html")

    # 2. baseball-data.com pages (2015-2026)
    for y in YEARS:
        yp = year_prefix(y)
        # Homepage for standings
        urls.append(f"https://baseball-data.com/{yp}")
        # Hitter stats (all players)
        urls.append(f"https://baseball-data.com/{yp}stats/hitter-all/avg-1.html")
        urls.append(f"https://baseball-data.com/{yp}stats/hitter2-all/avg-1.html")
        urls.append(f"https://baseball-data.com/{yp}stats/hitter3-all/avg-1.html")
        urls.append(f"https://baseball-data.com/{yp}stats/hitter4-all/avg-1.html")
        # Pitcher stats (all players)
        urls.append(f"https://baseball-data.com/{yp}stats/pitcher-all/era-1.html")
        urls.append(f"https://baseball-data.com/{yp}stats/pitcher2-all/era-1.html")
        urls.append(f"https://baseball-data.com/{yp}stats/pitcher3-all/era-1.html")
        # Team specific pages (players, lineups, position lineups)
        for tc in TEAM_CODES:
            urls.append(f"https://baseball-data.com/{yp}player/{tc}/")
            urls.append(f"https://baseball-data.com/{yp}lineup/{tc}.html")
            urls.append(f"https://baseball-data.com/{yp}lineup/{tc}-pos.html")
    return urls


def download_all():
    urls = collect_all_urls()
    print(f"Total URLs to fetch/check cache: {len(urls)}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_url, u): u for u in urls}
        done_cnt = 0
        for fut in as_completed(futures):
            done_cnt += 1
            if done_cnt % 50 == 0 or done_cnt == len(urls):
                print(f"Progress: {done_cnt}/{len(urls)} URLs fetched")


def clean_header(th_text: str) -> str:
    s = re.sub(r"[\s\n\r　]+", "", th_text)
    # Fix vertical bar in セーブ / ホールド / ボーク on baseball-data.com
    s = s.replace("セlブ", "セーブ").replace("ホlルド", "ホールド").replace("ボlク", "ボーク")
    return s


def parse_standings():
    rows = []
    for y in YEARS:
        yp = year_prefix(y)
        html = fetch_url(f"https://baseball-data.com/{yp}")
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        tables = soup.select("table.standings")
        # First two tables are セ・リーグ and パ・リーグ一軍
        for idx, league in enumerate(["CE", "PA"]):
            if idx >= len(tables):
                break
            tbl = tables[idx]
            for tr in tbl.find_all("tr")[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 8:
                    rank = int(tds[0]) if tds[0].isdigit() else np.nan
                    team = tds[1]
                    games = float(tds[2]) if tds[2] else np.nan
                    wins = float(tds[3]) if tds[3] else np.nan
                    losses = float(tds[4]) if tds[4] else np.nan
                    ties = float(tds[5]) if tds[5] else np.nan
                    win_pct_str = tds[6]
                    win_pct = float(win_pct_str) if win_pct_str.replace(".", "").isdigit() else np.nan
                    gb_str = tds[7]
                    gb = 0.0 if gb_str in ("優勝", "-") else (float(gb_str) if re.match(r"^-?\d+(\.\d+)?$", gb_str) else np.nan)
                    rows.append({
                        "year": y,
                        "league": league,
                        "team": team,
                        "team_rank": rank,
                        "team_games": games,
                        "team_wins": wins,
                        "team_losses": losses,
                        "team_ties": ties,
                        "team_win_pct": win_pct,
                        "team_gb": gb,
                        "is_champion": 1 if rank == 1 else 0,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_DIR, "standings.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved standings.csv: {len(df)} rows")
    return df


def parse_stats_table(html: str):
    if not html:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.find("table", id="tbl")
    if not tbl:
        return pd.DataFrame()
    thead = tbl.find("thead")
    headers = [clean_header(th.get_text()) for th in thead.find_all("th")]
    tbody = tbl.find("tbody")
    data = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != len(headers):
            continue
        row = {}
        for h, td in zip(headers, tds):
            val = td.get_text(strip=True)
            row[h] = val
            if h == "選手名":
                a = td.find("a")
                row["player_href"] = a["href"] if a and a.has_attr("href") else ""
        data.append(row)
    return pd.DataFrame(data)


def convert_ip_to_float(ip_val):
    """In baseball, 146.2 means 146 and 2/3 innings."""
    try:
        s = str(ip_val).strip()
        if "." in s:
            parts = s.split(".")
            whole = int(parts[0])
            frac = int(parts[1])
            return whole + (frac / 3.0)
        return float(s)
    except Exception:
        return np.nan


def parse_hitters():
    all_years = []
    for y in YEARS:
        yp = year_prefix(y)
        dfs = []
        for page in ["hitter-all", "hitter2-all", "hitter3-all", "hitter4-all"]:
            url = f"https://baseball-data.com/{yp}stats/{page}/avg-1.html"
            df_p = parse_stats_table(fetch_url(url))
            if not df_p.empty:
                if "順位" in df_p.columns:
                    df_p = df_p.drop(columns=["順位"])
                dfs.append(df_p)
        if not dfs:
            continue
        base = dfs[0]
        for df_next in dfs[1:]:
            key_cols = ["選手名", "チーム", "player_href"]
            overlap = [c for c in df_next.columns if c in base.columns and c not in key_cols]
            df_next_clean = df_next.drop(columns=overlap)
            base = pd.merge(base, df_next_clean, on=key_cols, how="outer")
        base["year"] = y
        base["league"] = base["チーム"].map(TEAM_TO_LEAGUE)
        base["norm_name"] = base["選手名"].apply(normalize_name)
        all_years.append(base)

    df = pd.concat(all_years, ignore_index=True)
    non_num = {"選手名", "チーム", "player_href", "year", "league", "norm_name"}
    for col in df.columns:
        if col not in non_num:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df.to_csv(os.path.join(PROCESSED_DIR, "hitters.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved hitters.csv: {len(df)} rows, {len(df.columns)} cols")
    return df


def parse_pitchers():
    all_years = []
    for y in YEARS:
        yp = year_prefix(y)
        dfs = []
        for page in ["pitcher-all", "pitcher2-all", "pitcher3-all"]:
            url = f"https://baseball-data.com/{yp}stats/{page}/era-1.html"
            df_p = parse_stats_table(fetch_url(url))
            if not df_p.empty:
                if "順位" in df_p.columns:
                    df_p = df_p.drop(columns=["順位"])
                dfs.append(df_p)
        if not dfs:
            continue
        base = dfs[0]
        for df_next in dfs[1:]:
            key_cols = ["選手名", "チーム", "player_href"]
            overlap = [c for c in df_next.columns if c in base.columns and c not in key_cols]
            df_next_clean = df_next.drop(columns=overlap)
            base = pd.merge(base, df_next_clean, on=key_cols, how="outer")
        base["year"] = y
        base["league"] = base["チーム"].map(TEAM_TO_LEAGUE)
        base["norm_name"] = base["選手名"].apply(normalize_name)
        all_years.append(base)

    df = pd.concat(all_years, ignore_index=True)
    non_num = {"選手名", "チーム", "player_href", "year", "league", "norm_name"}
    for col in df.columns:
        if col not in non_num:
            if col == "投球回":
                df["投球回_raw"] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].apply(convert_ip_to_float)
            else:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df.to_csv(os.path.join(PROCESSED_DIR, "pitchers.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved pitchers.csv: {len(df)} rows, {len(df.columns)} cols")
    return df


def parse_players_bio():
    rows = []
    for y in YEARS:
        yp = year_prefix(y)
        for tc, (team_name, league) in TEAM_CODES.items():
            url = f"https://baseball-data.com/{yp}player/{tc}/"
            html = fetch_url(url)
            if not html:
                continue
            soup = BeautifulSoup(html, "lxml")
            tbl = soup.find("table", id="tbl")
            if not tbl:
                continue
            tbody = tbl.find("tbody")
            if not tbody:
                continue
            for tr in tbody.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) < 12:
                    continue
                a = tr.find("a")
                href = a["href"] if a and a.has_attr("href") else ""
                p_name = tds[1]
                pos_reg = tds[2]
                age = pd.to_numeric(tds[4].replace("歳", ""), errors="coerce")
                pro_yr = pd.to_numeric(tds[5].replace("年", ""), errors="coerce")
                height = pd.to_numeric(tds[6].replace("cm", ""), errors="coerce")
                weight = pd.to_numeric(tds[7].replace("kg", ""), errors="coerce")
                throws_bats = tds[9]
                salary_str = tds[11].replace("万円", "").replace(",", "")
                salary = pd.to_numeric(salary_str, errors="coerce")
                rows.append({
                    "year": y,
                    "league": league,
                    "チーム": team_name,
                    "選手名": p_name,
                    "norm_name": normalize_name(p_name),
                    "player_href": href,
                    "登録守備": pos_reg,
                    "年齢": age,
                    "プロ年数": pro_yr,
                    "身長": height,
                    "体重": weight,
                    "投打": throws_bats,
                    "推定年俸": salary,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_DIR, "players_bio.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved players_bio.csv: {len(df)} rows")
    return df


def parse_lineups():
    pos_cols = ["投手", "捕手", "一塁手", "二塁手", "三塁手", "遊撃手", "左翼手", "中堅手", "右翼手", "指名打者"]
    records = {}  # key: (year, team_name, norm_name) -> dict of counts

    for y in YEARS:
        yp = year_prefix(y)
        for tc, (team_name, league) in TEAM_CODES.items():
            # 1. Position starts
            url_pos = f"https://baseball-data.com/{yp}lineup/{tc}-pos.html"
            html_pos = fetch_url(url_pos)
            if html_pos:
                soup = BeautifulSoup(html_pos, "lxml")
                tbl = soup.find("table", class_="lineup")
                if tbl:
                    trs = tbl.find_all("tr")
                    for tr in trs:
                        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                        if len(tds) < 11:
                            continue
                        # tds[0] is date, tds[1..10] are positions
                        for p_idx, p_col in enumerate(pos_cols):
                            raw_p = tds[p_idx + 1]
                            if not raw_p or raw_p == "-":
                                continue
                            nn = normalize_name(raw_p)
                            key = (y, team_name, nn)
                            if key not in records:
                                records[key] = {
                                    "year": y,
                                    "league": league,
                                    "チーム": team_name,
                                    "norm_name": nn,
                                    "raw_lineup_name": raw_p,
                                    **{f"スタメン_{c}": 0 for c in pos_cols},
                                    **{f"打順_{i}番": 0 for i in range(1, 10)},
                                }
                            records[key][f"スタメン_{p_col}"] += 1

            # 2. Batting order starts
            url_ord = f"https://baseball-data.com/{yp}lineup/{tc}.html"
            html_ord = fetch_url(url_ord)
            if html_ord:
                soup = BeautifulSoup(html_ord, "lxml")
                tbl = soup.find("table", class_="lineup")
                if tbl:
                    trs = tbl.find_all("tr")
                    for tr in trs:
                        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                        if len(tds) < 10:
                            continue
                        for b_order in range(1, 10):
                            raw_p = tds[b_order]
                            if not raw_p or raw_p == "-":
                                continue
                            nn = normalize_name(raw_p)
                            key = (y, team_name, nn)
                            if key not in records:
                                records[key] = {
                                    "year": y,
                                    "league": league,
                                    "チーム": team_name,
                                    "norm_name": nn,
                                    "raw_lineup_name": raw_p,
                                    **{f"スタメン_{c}": 0 for c in pos_cols},
                                    **{f"打順_{i}番": 0 for i in range(1, 10)},
                                }
                            records[key][f"打順_{b_order}番"] += 1

    df = pd.DataFrame(list(records.values()))
    df["スタメン_外野手"] = df["スタメン_左翼手"] + df["スタメン_中堅手"] + df["スタメン_右翼手"]
    df["スタメン_総数"] = df[[f"スタメン_{c}" for c in pos_cols]].sum(axis=1)
    # Calculate weighted average batting order
    order_weights = sum(df[f"打順_{i}番"] * i for i in range(1, 10))
    order_total = sum(df[f"打順_{i}番"] for i in range(1, 10))
    df["平均打順"] = np.where(order_total > 0, order_weights / order_total, np.nan)
    df["中軸スタメン数"] = df["打順_3番"] + df["打順_4番"] + df["打順_5番"]
    df["上位スタメン数"] = df["打順_1番"] + df["打順_2番"]

    df.to_csv(os.path.join(PROCESSED_DIR, "lineup_counts.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved lineup_counts.csv: {len(df)} rows")
    return df


def parse_best_nine():
    rows = []
    pos_map = {
        "投手": "投手",
        "捕手": "捕手",
        "一塁手": "一塁手",
        "二塁手": "二塁手",
        "三塁手": "三塁手",
        "遊撃手": "遊撃手",
        "外野手": "外野手",
        "ベストDH": "指名打者",
        "指名打者": "指名打者",
    }

    for y in AWARD_YEARS:
        url = f"https://npb.jp/award/{y}/voting_bt9.html"
        html = fetch_url(url)
        if not html:
            print(f"[ERROR] Missing Best Nine page for {y}")
            continue
        soup = BeautifulSoup(html, "lxml")

        # Extract valid votes for CE and PA
        valid_votes = {"CE": 270, "PA": 230}
        top_tbl = soup.select_one("div.table_center table")
        if top_tbl:
            for tr in top_tbl.find_all("tr"):
                th = tr.find("th")
                tds = tr.find_all("td")
                if th and "有効投票数" in th.get_text() and len(tds) >= 2:
                    v_ce = re.sub(r"\D", "", tds[0].get_text())
                    v_pa = re.sub(r"\D", "", tds[1].get_text())
                    if v_ce:
                        valid_votes["CE"] = int(v_ce)
                    if v_pa:
                        valid_votes["PA"] = int(v_pa)

        for div_cls, league in [("div.half_left", "CE"), ("div.half_right", "PA")]:
            sec = soup.select_one(div_cls)
            if not sec:
                continue
            tbl = sec.find("table")
            if not tbl:
                continue
            current_pos = None
            for tr in tbl.find_all("tr"):
                th = tr.find("th")
                if th and th.has_attr("colspan"):
                    pos_raw = th.get_text(strip=True)
                    current_pos = pos_map.get(pos_raw, pos_raw)
                    continue
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) == 4 and current_pos:
                    mark, p_name, team_paren, votes_str = tds
                    if not p_name or not votes_str:
                        continue
                    votes_clean = re.sub(r"\D", "", votes_str)
                    if not votes_clean:
                        continue
                    votes = int(votes_clean)
                    is_win = 1 if ("◎" in mark or "current" in tr.get("class", [])) else 0
                    t_short = re.sub(r"[（）()]", "", team_paren).strip()
                    team_full = NPB_TEAM_MAP.get(t_short, t_short)
                    rows.append({
                        "year": y,
                        "league": league,
                        "position": current_pos,
                        "npb_name": p_name,
                        "norm_name": normalize_name(p_name),
                        "チーム": team_full,
                        "votes": votes,
                        "valid_votes": valid_votes[league],
                        "vote_share": votes / valid_votes[league],
                        "is_winner": is_win,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_DIR, "best_nine_votes.csv"), index=False, encoding="utf-8-sig")
    winners = df[df["is_winner"] == 1]
    print(f"Saved best_nine_votes.csv: {len(df)} rows, Total winners (2015-2025): {len(winners)} (Expected: 11 * 19 = 209)")
    return df


if __name__ == "__main__":
    download_all()
    df_standings = parse_standings()
    df_hitters = parse_hitters()
    df_pitchers = parse_pitchers()
    df_bio = parse_players_bio()
    df_lineups = parse_lineups()
    df_b9 = parse_best_nine()
    print("All data parsing complete!")
