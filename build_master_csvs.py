import os
import gzip
import base64
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DIR = os.path.join(BASE_DIR, "web")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)


def build_master_tables():
    standings = pd.read_csv(os.path.join(PROCESSED_DIR, "standings.csv"))
    hitters = pd.read_csv(os.path.join(PROCESSED_DIR, "hitters.csv"))
    pitchers = pd.read_csv(os.path.join(PROCESSED_DIR, "pitchers.csv"))
    bio = pd.read_csv(os.path.join(PROCESSED_DIR, "players_bio.csv"))
    lineups = pd.read_csv(os.path.join(PROCESSED_DIR, "lineup_counts.csv"))
    b9 = pd.read_csv(os.path.join(PROCESSED_DIR, "best_nine_votes.csv"))

    # Rename standings columns to clear Japanese names
    st_clean = standings.rename(columns={
        "year": "年度",
        "league": "リーグ",
        "team": "チーム",
        "team_rank": "チーム順位",
        "team_games": "チーム試合数",
        "team_wins": "チーム勝利",
        "team_losses": "チーム敗北",
        "team_ties": "チーム引分",
        "team_win_pct": "チーム勝率",
        "team_gb": "ゲーム差",
        "is_champion": "リーグ優勝フラグ",
    })

    # Deduplicate bio per (year, league, チーム, norm_name)
    bio_clean = bio.drop_duplicates(["year", "league", "チーム", "norm_name"]).rename(columns={
        "year": "年度",
        "league": "リーグ",
        "推定年俸": "推定年俸_万円",
    })[["年度", "リーグ", "チーム", "norm_name", "登録守備", "年齢", "プロ年数", "身長", "体重", "投打", "推定年俸_万円"]]

    # Deduplicate lineups per (year, league, チーム, norm_name)
    pos_targets = ["捕手", "一塁手", "二塁手", "三塁手", "遊撃手", "外野手", "指名打者"]
    lin_clean = lineups.groupby(["year", "league", "チーム", "norm_name"], as_index=False).first().rename(columns={
        "year": "年度",
        "league": "リーグ",
    })
    # Determine primary starting position for fielders
    starts_mat = lin_clean[[f"スタメン_{p}" for p in pos_targets]].fillna(0).values
    max_idx = np.argmax(starts_mat, axis=1)
    max_val = np.max(starts_mat, axis=1)
    lin_clean["最多スタメン守備位置"] = [pos_targets[i] if v > 0 else "なし" for i, v in zip(max_idx, max_val)]
    lin_clean["最多スタメン守備試合数"] = max_val

    lin_keep_cols = (
        ["年度", "リーグ", "チーム", "norm_name", "最多スタメン守備位置", "最多スタメン守備試合数", "スタメン_総数"]
        + [f"スタメン_{p}" for p in ["投手", "捕手", "一塁手", "二塁手", "三塁手", "遊撃手", "外野手", "左翼手", "中堅手", "右翼手", "指名打者"]]
        + ["平均打順", "上位スタメン数", "中軸スタメン数"]
        + [f"打順_{i}番" for i in range(1, 10)]
    )
    lin_clean = lin_clean[lin_keep_cols]

    # Summarize Best Nine votes per player-season (Separately for Batter positions vs Pitcher)
    b9_bat = b9[b9["position"] != "投手"].copy()
    # Sort so winning position or highest-vote position comes first
    b9_bat = b9_bat.sort_values(["is_winner", "votes"], ascending=[False, False])
    b9_bat_agg = b9_bat.groupby(["year", "league", "チーム", "norm_name"], as_index=False).agg(
        ベストナイン受賞フラグ=("is_winner", "max"),
        ベストナイン得票部門=("position", lambda s: "/".join(dict.fromkeys(s))),
        ベストナイン最上位部門=("position", "first"),
        ベストナイン得票数合計=("votes", "sum"),
        ベストナイン最大部門得票数=("votes", "max"),
        ベストナイン有効投票数=("valid_votes", "first"),
        ベストナイン最大得票率=("vote_share", "max"),
    ).rename(columns={"year": "年度", "league": "リーグ"})

    b9_pit = b9[b9["position"] == "投手"].copy()
    b9_pit_agg = b9_pit.groupby(["year", "league", "チーム", "norm_name"], as_index=False).agg(
        ベストナイン受賞フラグ=("is_winner", "max"),
        ベストナイン得票数=("votes", "sum"),
        ベストナイン有効投票数=("valid_votes", "first"),
        ベストナイン得票率=("vote_share", "max"),
    ).rename(columns={"year": "年度", "league": "リーグ"})

    # =========================================================================
    # 1. Master Hitters Table (2015-2026)
    # =========================================================================
    df_h = hitters.rename(columns={"year": "年度", "league": "リーグ"}).copy()
    df_h = pd.merge(df_h, st_clean, on=["年度", "リーグ", "チーム"], how="left")
    df_h = pd.merge(df_h, bio_clean, on=["年度", "リーグ", "チーム", "norm_name"], how="left")
    df_h = pd.merge(df_h, lin_clean, on=["年度", "リーグ", "チーム", "norm_name"], how="left")
    df_h = pd.merge(df_h, b9_bat_agg, on=["年度", "リーグ", "チーム", "norm_name"], how="left")

    df_h["ベストナイン受賞フラグ"] = df_h["ベストナイン受賞フラグ"].fillna(0).astype(int)
    df_h["ベストナイン得票部門"] = df_h["ベストナイン得票部門"].fillna("")
    df_h["ベストナイン最上位部門"] = df_h["ベストナイン最上位部門"].fillna("")
    df_h["ベストナイン得票数合計"] = df_h["ベストナイン得票数合計"].fillna(0).astype(int)
    df_h["ベストナイン最大部門得票数"] = df_h["ベストナイン最大部門得票数"].fillna(0).astype(int)
    df_h["ベストナイン最大得票率"] = df_h["ベストナイン最大得票率"].fillna(0.0).round(4)

    for c in [col for col in lin_keep_cols if col.startswith("スタメン_") or col.startswith("打順_") or col in ("最多スタメン守備試合数", "上位スタメン数", "中軸スタメン数")]:
        df_h[c] = df_h[c].fillna(0).astype(int)
    df_h["最多スタメン守備位置"] = df_h["最多スタメン守備位置"].fillna("なし")
    df_h["平均打順"] = df_h["平均打順"].round(2)
    df_h["規定打席到達フラグ"] = (df_h["打席数"] >= np.round(df_h["チーム試合数"] * 3.1)).astype(int)

    # Order columns cleanly
    h_front_cols = [
        "年度", "リーグ", "チーム", "選手名", "登録守備", "最多スタメン守備位置", "最多スタメン守備試合数",
        "ベストナイン受賞フラグ", "ベストナイン得票部門", "ベストナイン最大部門得票数", "ベストナイン最大得票率",
        "チーム順位", "チーム勝率", "ゲーム差", "リーグ優勝フラグ",
        "年齢", "プロ年数", "身長", "体重", "投打", "推定年俸_万円", "規定打席到達フラグ",
        "打率", "試合", "打席数", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打", "塁打", "打点", "盗塁", "盗塁刺",
        "犠打", "犠飛", "四球", "敬遠", "死球", "四死球", "三振", "併殺打",
        "出塁率", "長打率", "OPS", "NOI", "GPA", "IsoP", "IsoD", "RC", "RC27", "XR", "XR27",
        "BABIP", "SecA", "TA", "本塁打率", "三振率", "四球率", "PSN", "BB/K",
        "スタメン_総数", "スタメン_捕手", "スタメン_一塁手", "スタメン_二塁手", "スタメン_三塁手", "スタメン_遊撃手",
        "スタメン_外野手", "スタメン_左翼手", "スタメン_中堅手", "スタメン_右翼手", "スタメン_指名打者", "スタメン_投手",
        "平均打順", "上位スタメン数", "中軸スタメン数",
        "打順_1番", "打順_2番", "打順_3番", "打順_4番", "打順_5番", "打順_6番", "打順_7番", "打順_8番", "打順_9番",
    ]
    h_cols = [c for c in h_front_cols if c in df_h.columns]
    df_h = df_h[h_cols].sort_values(["年度", "リーグ", "チーム順位", "打席数"], ascending=[True, True, True, False])

    hitter_csv_path = os.path.join(OUTPUT_DIR, "master_hitters_2015_2026.csv")
    df_h.to_csv(hitter_csv_path, index=False, encoding="utf-8-sig")
    df_h.to_csv(os.path.join(WEB_DIR, "master_hitters_2015_2026.csv"), index=False, encoding="utf-8-sig")

    # =========================================================================
    # 2. Master Pitchers Table (2015-2026)
    # =========================================================================
    df_p = pitchers.rename(columns={"year": "年度", "league": "リーグ"}).copy()
    df_p["投球回_換算値"] = df_p["投球回"].round(2)
    if "投球回_raw" in df_p.columns:
        df_p["投球回_表記"] = df_p["投球回_raw"]
    df_p = pd.merge(df_p, st_clean, on=["年度", "リーグ", "チーム"], how="left")
    df_p = pd.merge(df_p, bio_clean, on=["年度", "リーグ", "チーム", "norm_name"], how="left")
    df_p = pd.merge(df_p, lin_clean[["年度", "リーグ", "チーム", "norm_name", "スタメン_投手"]].rename(columns={"スタメン_投手": "先発登板数"}), on=["年度", "リーグ", "チーム", "norm_name"], how="left")
    df_p = pd.merge(df_p, b9_pit_agg, on=["年度", "リーグ", "チーム", "norm_name"], how="left")

    df_p["先発登板数"] = df_p["先発登板数"].fillna(0).astype(int)
    df_p["ベストナイン受賞フラグ"] = df_p["ベストナイン受賞フラグ"].fillna(0).astype(int)
    df_p["ベストナイン得票数"] = df_p["ベストナイン得票数"].fillna(0).astype(int)
    df_p["ベストナイン得票率"] = df_p["ベストナイン得票率"].fillna(0.0).round(4)
    df_p["規定投球回到達フラグ"] = (df_p["投球回_換算値"] >= df_p["チーム試合数"]).astype(int)

    p_front_cols = [
        "年度", "リーグ", "チーム", "選手名", "登録守備", "先発登板数",
        "ベストナイン受賞フラグ", "ベストナイン得票数", "ベストナイン得票率",
        "チーム順位", "チーム勝率", "ゲーム差", "リーグ優勝フラグ",
        "年齢", "プロ年数", "身長", "体重", "投打", "推定年俸_万円", "規定投球回到達フラグ",
        "防御率", "試合", "勝利", "敗北", "セーブ", "ホールド", "HP", "完投", "完封勝", "無四球", "勝率",
        "打者", "投球回_表記", "投球回_換算値", "被安打", "被本塁打", "与四球", "敬遠", "与死球", "与四死球",
        "奪三振", "暴投", "ボーク", "失点", "自責点",
        "WHIP", "DIPS", "被安打率", "被本打率", "奪三振率", "与四死率", "K/BB", "HldR", "PFR",
    ]
    p_cols = [c for c in p_front_cols if c in df_p.columns]
    df_p = df_p[p_cols].sort_values(["年度", "リーグ", "チーム順位", "投球回_換算値"], ascending=[True, True, True, False])

    pitcher_csv_path = os.path.join(OUTPUT_DIR, "master_pitchers_2015_2026.csv")
    df_p.to_csv(pitcher_csv_path, index=False, encoding="utf-8-sig")
    df_p.to_csv(os.path.join(WEB_DIR, "master_pitchers_2015_2026.csv"), index=False, encoding="utf-8-sig")

    # =========================================================================
    # 3. Single Unified All-in-One Table (Hitters + Pitchers merged by player-season)
    # =========================================================================
    key_u = ["年度", "リーグ", "チーム", "選手名"]
    shared_meta = [
        "登録守備", "チーム順位", "チーム勝率", "ゲーム差", "リーグ優勝フラグ",
        "年齢", "プロ年数", "身長", "体重", "投打", "推定年俸_万円"
    ]
    df_h_pref = df_h.rename(columns={
        c: f"打者_{c}" for c in df_h.columns if c not in key_u + shared_meta and not c.startswith("スタメン_") and not c.startswith("打順_")
    })
    df_p_pref = df_p.rename(columns={
        c: f"投手_{c}" for c in df_p.columns if c not in key_u + shared_meta
    })
    # Drop duplicate shared_meta from df_p_pref before outer join, then coalesce
    df_all = pd.merge(df_h_pref, df_p_pref, on=key_u, how="outer", suffixes=("", "_pitmeta"))
    for sm in shared_meta:
        if f"{sm}_pitmeta" in df_all.columns:
            df_all[sm] = df_all[sm].combine_first(df_all[f"{sm}_pitmeta"])
            df_all = df_all.drop(columns=[f"{sm}_pitmeta"])

    df_all["ベストナイン受賞フラグ_総合"] = (
        df_all["打者_ベストナイン受賞フラグ"].fillna(0) + df_all["投手_ベストナイン受賞フラグ"].fillna(0)
    ).clip(0, 1).astype(int)

    all_csv_path = os.path.join(OUTPUT_DIR, "master_all_players_unified_2015_2026.csv")
    df_all.to_csv(all_csv_path, index=False, encoding="utf-8-sig")
    df_all.to_csv(os.path.join(WEB_DIR, "master_all_players_unified_2015_2026.csv"), index=False, encoding="utf-8-sig")

    print(f"Created master_hitters_2015_2026.csv: {df_h.shape}")
    print(f"Created master_pitchers_2015_2026.csv: {df_p.shape}")
    print(f"Created master_all_players_unified_2015_2026.csv: {df_all.shape}")

    # Compress to gzip + base64 so we can embed direct 1-click download into the HTML pages!
    payloads = {}
    for name, path in [
        ("hitters", hitter_csv_path),
        ("pitchers", pitcher_csv_path),
        ("all_unified", all_csv_path),
    ]:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        gz_bytes = gzip.compress(raw_bytes, compresslevel=9)
        b64_str = base64.b64encode(gz_bytes).decode("ascii")
        payloads[name] = {
            "filename": os.path.basename(path),
            "raw_kb": round(len(raw_bytes) / 1024, 1),
            "gz_kb": round(len(gz_bytes) / 1024, 1),
            "b64": b64_str,
        }
        print(f"  {name}: raw={payloads[name]['raw_kb']} KB -> gz={payloads[name]['gz_kb']} KB")

    with open(os.path.join(OUTPUT_DIR, "csv_b64_payloads.json"), "w", encoding="utf-8") as f:
        json.dump(payloads, f)


if __name__ == "__main__":
    build_master_tables()
