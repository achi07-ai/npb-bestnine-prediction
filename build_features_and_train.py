import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Configure Japanese font for matplotlib on Windows
plt.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean_data():
    standings = pd.read_csv(os.path.join(PROCESSED_DIR, "standings.csv"))
    hitters = pd.read_csv(os.path.join(PROCESSED_DIR, "hitters.csv"))
    pitchers = pd.read_csv(os.path.join(PROCESSED_DIR, "pitchers.csv"))
    bio = pd.read_csv(os.path.join(PROCESSED_DIR, "players_bio.csv"))
    lineups = pd.read_csv(os.path.join(PROCESSED_DIR, "lineup_counts.csv"))
    b9 = pd.read_csv(os.path.join(PROCESSED_DIR, "best_nine_votes.csv"))

    # Fix any DH position label variants in b9
    def clean_pos(p):
        p = str(p)
        if "DH" in p or "ＤＨ" in p or "指名打者" in p:
            return "指名打者"
        return p.strip()

    b9["position"] = b9["position"].apply(clean_pos)
    b9.to_csv(os.path.join(PROCESSED_DIR, "best_nine_votes.csv"), index=False, encoding="utf-8-sig")
    return standings, hitters, pitchers, bio, lineups, b9


def build_pitcher_dataset(standings, pitchers, bio, lineups, b9):
    df = pitchers.copy()

    # Merge standings
    df = pd.merge(
        df,
        standings[["year", "league", "team", "team_rank", "team_games", "team_wins", "team_losses", "team_win_pct", "team_gb", "is_champion"]],
        left_on=["year", "league", "チーム"],
        right_on=["year", "league", "team"],
        how="left",
    )

    # Merge lineup starts for pitchers (スタメン_投手 = games started)
    lin_p = lineups[["year", "league", "チーム", "norm_name", "スタメン_投手"]].copy()
    lin_p = lin_p.groupby(["year", "league", "チーム", "norm_name"], as_index=False).sum()
    df = pd.merge(df, lin_p, on=["year", "league", "チーム", "norm_name"], how="left")
    df["スタメン_投手"] = df["スタメン_投手"].fillna(0)

    # Merge bio
    bio_p = bio[["year", "league", "チーム", "norm_name", "年齢", "プロ年数", "推定年俸"]].drop_duplicates(["year", "league", "チーム", "norm_name"])
    df = pd.merge(df, bio_p, on=["year", "league", "チーム", "norm_name"], how="left")

    # Merge Best Nine votes for position == '投手'
    b9_p = b9[b9["position"] == "投手"][["year", "league", "チーム", "norm_name", "votes", "valid_votes", "vote_share", "is_winner"]].copy()
    b9_p = b9_p.groupby(["year", "league", "チーム", "norm_name"], as_index=False).agg({
        "votes": "sum",
        "valid_votes": "first",
        "vote_share": "sum",
        "is_winner": "max",
    })
    df = pd.merge(df, b9_p, on=["year", "league", "チーム", "norm_name"], how="left")
    df["votes"] = df["votes"].fillna(0)
    df["vote_share"] = df["vote_share"].fillna(0.0)
    df["is_winner"] = df["is_winner"].fillna(0).astype(int)
    df["position"] = "投手"

    # Previous year Best Nine vote share & winner status (lag 1 year by norm_name)
    b9_p_prev = b9_p.groupby(["year", "norm_name"], as_index=False).agg({
        "vote_share": "max",
        "is_winner": "max",
    }).rename(columns={"vote_share": "prev_vote_share", "is_winner": "prev_is_winner"})
    b9_p_prev["year"] = b9_p_prev["year"] + 1
    df = pd.merge(df, b9_p_prev, on=["year", "norm_name"], how="left")
    df["prev_vote_share"] = df["prev_vote_share"].fillna(0.0)
    df["prev_is_winner"] = df["prev_is_winner"].fillna(0).astype(int)

    # Domain engineered features for Pitchers
    df["貯金"] = df["勝利"] - df["敗北"]
    df["先発数"] = df["スタメン_投手"]
    df["先発比率"] = np.where(df["試合"] > 0, df["先発数"] / df["試合"], 0.0)
    df["規定投球回到達"] = (df["投球回"] >= df["team_games"]).astype(int)
    df["規定投球回充足率"] = df["投球回"] / df["team_games"]
    df["QS相当指標"] = df["投球回"] / np.maximum(df["先発数"], 1)
    df["log_推定年俸"] = np.log1p(df["推定年俸"].fillna(0))
    df["is_rookie_regular"] = ((df["プロ年数"].fillna(99) == 1) & (df["規定投球回到達"] == 1)).astype(int)

    # League-year relative features
    def add_league_features(g):
        g = g.copy()
        # Ranks (1 is best)
        for col in ["勝利", "奪三振", "投球回", "勝率", "完投", "完封勝", "貯金", "セーブ", "HP", "先発数"]:
            g[f"lg_rank_{col}"] = g[col].rank(ascending=False, method="min")
            max_v = g[col].max()
            g[f"lg_diff_top_{col}"] = g[col] - max_v
            g[f"lg_ratio_top_{col}"] = np.where(max_v > 0, g[col] / max_v, 0.0)

        # Qualified pitcher ERA / WHIP / DIPS ranks
        qual_mask = g["規定投球回到達"] == 1
        for col in ["防御率", "WHIP", "DIPS"]:
            qual_vals = g.loc[qual_mask, col]
            min_qual = qual_vals.min() if len(qual_vals) > 0 else g[col].min()
            # Rank among qualified (unqualified get rank 50+)
            q_rank = g.loc[qual_mask, col].rank(ascending=True, method="min")
            g[f"lg_qual_rank_{col}"] = 99.0
            g.loc[qual_mask, f"lg_qual_rank_{col}"] = q_rank
            g[f"lg_diff_best_qual_{col}"] = g[col] - min_qual

        # Title flags
        g["title_最多勝"] = (g["lg_rank_勝利"] == 1).astype(int)
        g["title_最優秀防御率"] = (g["lg_qual_rank_防御率"] == 1).astype(int)
        g["title_最多奪三振"] = (g["lg_rank_奪三振"] == 1).astype(int)
        # Winning pct title usually requires >= 13 wins
        w13 = g["勝利"] >= 13
        if w13.any():
            max_wp = g.loc[w13, "勝率"].max()
            g["title_最高勝率"] = ((g["勝利"] >= 13) & (g["勝率"] == max_wp)).astype(int)
        else:
            g["title_最高勝率"] = 0
        g["title_最多セーブ"] = (g["lg_rank_セーブ"] == 1).astype(int)
        g["主要タイトル数_先発"] = g["title_最多勝"] + g["title_最優秀防御率"] + g["title_最多奪三振"] + g["title_最高勝率"]

        # Interactions with team rank / champion
        g["勝利_x_優勝"] = g["勝利"] * g["is_champion"]
        g["貯金_x_チーム勝率"] = g["貯金"] * g["team_win_pct"]
        g["先発勝利インパクト"] = (g["勝利"] * 1.5 + g["貯金"] * 1.0 - g["防御率"] * 3.0) * g["規定投球回到達"]
        return g

    df = pd.concat([add_league_features(g) for _, g in df.groupby(["year", "league"])], ignore_index=True)
    df["league_id"] = (df["league"] == "PA").astype(int)
    return df


def build_batter_dataset(standings, hitters, bio, lineups, b9):
    """
    Constructs candidate (year, league, position, player) rows for all 7 batter positions:
    CE: 捕手, 一塁手, 二塁手, 三塁手, 遊撃手, 外野手
    PA: 捕手, 一塁手, 二塁手, 三塁手, 遊撃手, 外野手, 指名打者
    """
    df_h = hitters.copy()

    # Merge standings
    df_h = pd.merge(
        df_h,
        standings[["year", "league", "team", "team_rank", "team_games", "team_wins", "team_losses", "team_win_pct", "team_gb", "is_champion"]],
        left_on=["year", "league", "チーム"],
        right_on=["year", "league", "team"],
        how="left",
    )

    # Merge bio
    bio_b = bio[["year", "league", "チーム", "norm_name", "登録守備", "年齢", "プロ年数", "推定年俸"]].drop_duplicates(["year", "league", "チーム", "norm_name"])
    df_h = pd.merge(df_h, bio_b, on=["year", "league", "チーム", "norm_name"], how="left")

    # Merge lineup counts
    pos_cols = ["捕手", "一塁手", "二塁手", "三塁手", "遊撃手", "外野手", "指名打者", "左翼手", "中堅手", "右翼手"]
    lin_cols = ["year", "league", "チーム", "norm_name", "スタメン_総数", "平均打順", "中軸スタメン数", "上位スタメン数"] + [f"スタメン_{c}" for c in pos_cols]
    lin_b = lineups[lin_cols].groupby(["year", "league", "チーム", "norm_name"], as_index=False).first()
    df_h = pd.merge(df_h, lin_b, on=["year", "league", "チーム", "norm_name"], how="left")
    for c in [f"スタメン_{p}" for p in pos_cols] + ["スタメン_総数", "中軸スタメン数", "上位スタメン数"]:
        df_h[c] = df_h[c].fillna(0)
    df_h["平均打順"] = df_h["平均打順"].fillna(8.0)

    # Basic engineered batter features
    df_h["規定打席数"] = np.round(df_h["team_games"] * 3.1)
    df_h["規定打席到達"] = (df_h["打席数"] >= df_h["規定打席数"]).astype(int)
    df_h["規定打席充足率"] = df_h["打席数"] / df_h["規定打席数"]
    df_h["出場試合率"] = df_h["試合"] / df_h["team_games"]
    df_h["スタメン出場率"] = df_h["スタメン_総数"] / df_h["team_games"]
    df_h["長打数"] = df_h["二塁打"] + df_h["三塁打"] + df_h["本塁打"]
    df_h["log_推定年俸"] = np.log1p(df_h["推定年俸"].fillna(0))

    # Determine each player's primary and secondary starting position
    target_positions = ["捕手", "一塁手", "二塁手", "三塁手", "遊撃手", "外野手", "指名打者"]
    starts_mat = df_h[[f"スタメン_{p}" for p in target_positions]].values
    max_pos_idx = np.argmax(starts_mat, axis=1)
    max_starts = np.max(starts_mat, axis=1)
    df_h["primary_pos"] = [target_positions[i] if m > 0 else "" for i, m in zip(max_pos_idx, max_starts)]
    df_h["primary_pos_starts"] = max_starts

    # League-wide batter ranks and titles
    def add_batter_league_features(g):
        g = g.copy()
        qual_mask = g["規定打席到達"] == 1
        for col in ["本塁打", "打点", "安打", "得点", "塁打", "盗塁", "四球", "RC", "XR", "打席数"]:
            g[f"lg_rank_{col}"] = g[col].rank(ascending=False, method="min")
            max_v = g[col].max()
            g[f"lg_diff_top_{col}"] = g[col] - max_v
            g[f"lg_ratio_top_{col}"] = np.where(max_v > 0, g[col] / max_v, 0.0)

        for col in ["打率", "出塁率", "長打率", "OPS", "RC27", "XR27", "NOI", "GPA", "IsoP"]:
            # Among qualified batters
            qual_vals = g.loc[qual_mask, col]
            max_qual = qual_vals.max() if len(qual_vals) > 0 else g[col].max()
            mean_qual = qual_vals.mean() if len(qual_vals) > 0 else g[col].mean()
            std_qual = qual_vals.std() if len(qual_vals) > 1 else 1.0
            g[f"lg_qual_rank_{col}"] = 99.0
            g.loc[qual_mask, f"lg_qual_rank_{col}"] = g.loc[qual_mask, col].rank(ascending=False, method="min")
            g[f"lg_diff_top_qual_{col}"] = g[col] - max_qual
            g[f"lg_z_qual_{col}"] = (g[col] - mean_qual) / (std_qual if std_qual > 0 else 1.0)

        g["title_首位打者"] = (g["lg_qual_rank_打率"] == 1).astype(int)
        g["title_本塁打王"] = (g["lg_rank_本塁打"] == 1).astype(int)
        g["title_打点王"] = (g["lg_rank_打点"] == 1).astype(int)
        g["title_最多安打"] = (g["lg_rank_安打"] == 1).astype(int)
        g["title_最高出塁率"] = (g["lg_qual_rank_出塁率"] == 1).astype(int)
        g["title_最多盗塁"] = (g["lg_rank_盗塁"] == 1).astype(int)
        g["打撃タイトル数"] = (
            g["title_首位打者"] + g["title_本塁打王"] + g["title_打点王"]
            + g["title_最多安打"] + g["title_最高出塁率"] + g["title_最多盗塁"]
        )
        return g

    df_h = pd.concat([add_batter_league_features(g) for _, g in df_h.groupby(["year", "league"])], ignore_index=True)

    # Expand to candidate (year, league, position, player) rows
    # A player is a candidate for `pos` if they started >= 1 game at `pos` OR got votes at `pos`
    b9_b = b9[b9["position"] != "投手"][["year", "league", "position", "チーム", "norm_name", "votes", "valid_votes", "vote_share", "is_winner"]].copy()
    b9_b = b9_b.groupby(["year", "league", "position", "チーム", "norm_name"], as_index=False).agg({
        "votes": "sum",
        "valid_votes": "first",
        "vote_share": "sum",
        "is_winner": "max",
    })

    cand_list = []
    for pos in target_positions:
        sub = df_h.copy()
        if pos == "指名打者":
            sub = sub[sub["league"] == "PA"].copy()
        sub["position"] = pos
        sub["pos_starts"] = sub[f"スタメン_{pos}"]
        # Merge votes for this specific position
        sub = pd.merge(
            sub,
            b9_b[b9_b["position"] == pos][["year", "league", "チーム", "norm_name", "votes", "valid_votes", "vote_share", "is_winner"]],
            on=["year", "league", "チーム", "norm_name"],
            how="left",
        )
        sub["votes"] = sub["votes"].fillna(0)
        sub["vote_share"] = sub["vote_share"].fillna(0.0)
        sub["is_winner"] = sub["is_winner"].fillna(0).astype(int)

        # Filter to players who started >= 5 games at this position OR received votes
        sub = sub[(sub["pos_starts"] >= 5) | (sub["votes"] > 0)].copy()
        cand_list.append(sub)

    df_cand = pd.concat(cand_list, ignore_index=True)

    # Merge previous year Best Nine vote share (same position & any position)
    b9_b_prev_pos = b9_b.groupby(["year", "position", "norm_name"], as_index=False).agg({
        "vote_share": "max",
        "is_winner": "max",
    }).rename(columns={"vote_share": "prev_vote_share", "is_winner": "prev_is_winner"})
    b9_b_prev_pos["year"] = b9_b_prev_pos["year"] + 1
    df_cand = pd.merge(df_cand, b9_b_prev_pos, on=["year", "position", "norm_name"], how="left")
    df_cand["prev_vote_share"] = df_cand["prev_vote_share"].fillna(0.0)
    df_cand["prev_is_winner"] = df_cand["prev_is_winner"].fillna(0).astype(int)

    b9_b_prev_any = b9_b.groupby(["year", "norm_name"], as_index=False).agg({
        "vote_share": "max",
    }).rename(columns={"vote_share": "prev_any_pos_vote_share"})
    b9_b_prev_any["year"] = b9_b_prev_any["year"] + 1
    df_cand = pd.merge(df_cand, b9_b_prev_any, on=["year", "norm_name"], how="left")
    df_cand["prev_any_pos_vote_share"] = df_cand["prev_any_pos_vote_share"].fillna(0.0)

    # Position-specific features
    df_cand["pos_starts_rate_team"] = df_cand["pos_starts"] / df_cand["team_games"]
    df_cand["pos_starts_share_player"] = np.where(df_cand["スタメン_総数"] > 0, df_cand["pos_starts"] / df_cand["スタメン_総数"], 0.0)
    df_cand["is_primary_pos"] = (df_cand["position"] == df_cand["primary_pos"]).astype(int)
    df_cand["diff_from_primary_starts"] = df_cand["pos_starts"] - df_cand["primary_pos_starts"]
    df_cand["is_outfield"] = (df_cand["position"] == "外野手").astype(int)
    df_cand["slots_available"] = np.where(df_cand["position"] == "外野手", 3, 1)

    # 1) 新人レギュラーフラグ (Rookie full-time starter flag)
    df_cand["is_rookie"] = (df_cand["プロ年数"].fillna(99) == 1).astype(int)
    df_cand["is_rookie_regular"] = (
        (df_cand["プロ年数"].fillna(99) == 1)
        & (df_cand["pos_starts_rate_team"] >= 0.65)
        & (df_cand["is_primary_pos"] == 1)
    ).astype(int)

    # 2) 主砲の兼任割引フロア制限 (Cleanup/Mainstay multi-position share floor):
    # If a player starts >= 68% of team games (~98+ games) at their primary position AND starts >= 90% of team games overall
    # (or has >= 60 middle-of-the-order cleanup starts), playing additional games at a secondary position for team needs
    # should not overly penalize their primary position share below 0.90.
    is_mainstay_primary = (
        (df_cand["is_primary_pos"] == 1)
        & (df_cand["pos_starts_rate_team"] >= 0.68)
        & ((df_cand["スタメン出場率"] >= 0.90) | (df_cand["中軸スタメン数"] >= 60))
    )
    df_cand["eff_pos_share"] = np.where(
        is_mainstay_primary,
        np.maximum(df_cand["pos_starts_share_player"], 0.90),
        df_cand["pos_starts_share_player"],
    )

    # Effective offensive metrics weighted by playing time at the position (using eff_pos_share)
    pos_weight = np.clip(df_cand["pos_starts"] / (df_cand["team_games"] * 0.75), 0.0, 1.15) * (0.3 + 0.7 * df_cand["eff_pos_share"])
    df_cand["pos_weighted_OPS"] = df_cand["OPS"] * pos_weight
    df_cand["pos_weighted_RC"] = df_cand["RC"] * np.clip(df_cand["pos_starts"] / (df_cand["team_games"] * 0.75), 0.0, 1.10) * df_cand["eff_pos_share"]
    df_cand["pos_weighted_HR"] = df_cand["本塁打"] * df_cand["eff_pos_share"]
    df_cand["pos_weighted_RBI"] = df_cand["打点"] * df_cand["eff_pos_share"]
    df_cand["pos_weighted_H"] = df_cand["安打"] * np.clip(df_cand["pos_starts"] / (df_cand["team_games"] * 0.75), 0.0, 1.10) * df_cand["eff_pos_share"]
    df_cand["pos_weighted_AVG"] = df_cand["打率"] * pos_weight

    # Within (year, league, position) competition features!
    def add_position_group_features(g):
        g = g.copy()
        pos_name = g["position"].iloc[0]
        k_target = 3 if pos_name == "外野手" else 1

        # Define "regulars" at this position (primary position and >= 45% of team games started at this pos)
        reg_mask = (g["is_primary_pos"] == 1) & (g["pos_starts_rate_team"] >= 0.45)
        if reg_mask.sum() < k_target + 1:
            reg_mask = (g["is_primary_pos"] == 1) & (g["pos_starts_rate_team"] >= 0.30)
        if reg_mask.sum() < k_target + 1:
            reg_mask = g["pos_starts_rate_team"] >= 0.20
        if reg_mask.sum() < k_target:
            reg_mask = pd.Series(True, index=g.index)


        for col in [
            "pos_starts", "pos_weighted_OPS", "pos_weighted_RC", "pos_weighted_HR",
            "pos_weighted_RBI", "pos_weighted_H", "pos_weighted_AVG",
            "OPS", "RC", "RC27", "XR", "打率", "本塁打", "打点", "安打", "出塁率", "長打率", "打席数"
        ]:
            # Rank among all candidates at this position
            g[f"pos_rank_{col}"] = g[col].rank(ascending=False, method="min")
            # Rank among regulars at this position
            g[f"pos_reg_rank_{col}"] = 99.0
            g.loc[reg_mask, f"pos_reg_rank_{col}"] = g.loc[reg_mask, col].rank(ascending=False, method="min")

            # Compare against the K-th best regular (1st for IF/C/DH, 3rd for OF) and (K+1)-th best regular
            reg_vals = np.sort(g.loc[reg_mask, col].dropna().values)[::-1]
            top1_val = reg_vals[0] if len(reg_vals) >= 1 else 0.0
            kth_val = reg_vals[min(k_target - 1, len(reg_vals) - 1)] if len(reg_vals) >= 1 else 0.0
            kplus1_val = reg_vals[min(k_target, len(reg_vals) - 1)] if len(reg_vals) >= 1 else 0.0

            g[f"pos_diff_top1_{col}"] = g[col] - top1_val
            # Margin over the cutoff (vs 2nd best for single-slot, vs 4th best for Outfield!)
            # Notice: for the #1 player in a single-slot position, their margin over #2 is (val - kplus1_val)!
            # For other players, their deficit to #1 is (val - kth_val)!
            is_in_top_k = g[f"pos_reg_rank_{col}"] <= k_target
            g[f"pos_margin_cutoff_{col}"] = np.where(is_in_top_k, g[col] - kplus1_val, g[col] - kth_val)

            mean_v = g.loc[reg_mask, col].mean()
            std_v = g.loc[reg_mask, col].std()
            g[f"pos_z_{col}"] = (g[col] - mean_v) / (std_v if pd.notna(std_v) and std_v > 0 else 1.0)

        g["OPS_x_優勝"] = g["OPS"] * g["is_champion"]
        g["RC_x_優勝"] = g["RC"] * g["is_champion"]
        return g

    df_cand = pd.concat([add_position_group_features(g) for _, g in df_cand.groupby(["year", "league", "position"])], ignore_index=True)

    # Encode position as integer ID for LightGBM categorical feature
    pos_to_id = {p: i for i, p in enumerate(target_positions)}
    df_cand["position_id"] = df_cand["position"].map(pos_to_id)
    df_cand["league_id"] = (df_cand["league"] == "PA").astype(int)

    return df_cand


if __name__ == "__main__":
    standings, hitters, pitchers, bio, lineups, b9 = load_and_clean_data()
    df_pit = build_pitcher_dataset(standings, pitchers, bio, lineups, b9)
    df_bat = build_batter_dataset(standings, hitters, bio, lineups, b9)

    df_pit.to_csv(os.path.join(PROCESSED_DIR, "features_pitchers.csv"), index=False, encoding="utf-8-sig")
    df_bat.to_csv(os.path.join(PROCESSED_DIR, "features_batters.csv"), index=False, encoding="utf-8-sig")

    pit_w = df_pit[df_pit["year"] <= 2025]["is_winner"].sum()
    bat_w = df_bat[df_bat["year"] <= 2025]["is_winner"].sum()
    print(f"Pitchers dataset: {df_pit.shape}, Winners (2015-2025): {pit_w} (Expected: 22)")
    print(f"Batters dataset:  {df_bat.shape}, Winners (2015-2025): {bat_w} (Expected: 187)")
