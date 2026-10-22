import os
import pandas as pd
from train_models import train_pitcher_lgbm, train_batter_lgbm, predict_pitcher_lgbm, predict_batter_lgbm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

dp = pd.read_csv(os.path.join(PROCESSED_DIR, "features_pitchers.csv"))
db = pd.read_csv(os.path.join(PROCESSED_DIR, "features_batters.csv"))

# Same 9yr Train+Val model used for Test (2024-2025) evaluation in train_models.py
pit_tr = dp[dp["year"].between(2015, 2021)].copy()
pit_va = dp[dp["year"].between(2022, 2023)].copy()
bat_tr = db[db["year"].between(2015, 2021)].copy()
bat_va = db[db["year"].between(2022, 2023)].copy()
pm_va = train_pitcher_lgbm(pit_tr, pit_va)
bm_va = train_batter_lgbm(bat_tr, bat_va)

pit_trva = dp[dp["year"].between(2015, 2023)].copy()
bat_trva = db[db["year"].between(2015, 2023)].copy()
pm = train_pitcher_lgbm(pit_trva, None, params_override={"n_estimators": max(pm_va[0].best_iteration_ or 150, 100)})
bm = train_batter_lgbm(bat_trva, None, params_override={"n_estimators": max(bm_va[0].best_iteration_ or 200, 120)})

pte = dp[dp["year"].between(2024, 2025)].copy()
bte = db[db["year"].between(2024, 2025)].copy()
pte["score"], pte["prob"], pte["vsh"], pte["rnk"] = predict_pitcher_lgbm(pm, pte, weights=(0.75, 0.0, 0.25))
bte["score"], bte["prob"], bte["vsh"], bte["rnk"] = predict_batter_lgbm(bm, bte, weights=(0.15, 0.55, 0.30))

lines = []
lines.append("=== 2024 PA PITCHER ===")
sub_p = pte[(pte["year"] == 2024) & (pte["league"] == "PA")].sort_values("score", ascending=False)
cols_p = ["チーム", "選手名", "勝利", "敗北", "勝率", "防御率", "投球回", "奪三振", "prev_vote_share", "主要タイトル数_先発", "team_rank", "votes", "vote_share", "prob", "vsh", "rnk", "score", "is_winner"]
lines.append(sub_p[cols_p].head(6).to_string(index=False))

for y, lg, pos in [
    (2024, "CE", "一塁手"),
    (2024, "CE", "二塁手"),
    (2024, "PA", "捕手"),
    (2024, "PA", "外野手"),
    (2025, "PA", "三塁手"),
    (2025, "PA", "遊撃手"),
]:
    lines.append(f"\n=== {y} {lg} {pos} ===")
    sub_b = bte[(bte["year"] == y) & (bte["league"] == lg) & (bte["position"] == pos)].sort_values("score", ascending=False)
    cols_b = [
        "チーム", "選手名", "打率", "本塁打", "打点", "安打", "OPS", "RC",
        "pos_starts", "スタメン_総数", "eff_pos_share", "pos_weighted_RC", "pos_margin_cutoff_pos_weighted_RC",
        "prev_vote_share", "is_rookie_regular", "打撃タイトル数", "team_rank", "votes", "vote_share", "prob", "vsh", "rnk", "score", "is_winner"
    ]
    lines.append(sub_b[cols_b].head(6).to_string(index=False))

with open(os.path.join(OUTPUT_DIR, "error_analysis_2024_2025.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved error_analysis_2024_2025.txt")
