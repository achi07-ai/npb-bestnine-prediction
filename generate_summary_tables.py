import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACT_DIR = r"C:\Users\akats\.gemini\antigravity\brain\e3933ebd-d891-42c9-86c4-f190bddb31a2"

pit = pd.read_csv(os.path.join(PROCESSED_DIR, "features_pitchers.csv"))
bat = pd.read_csv(os.path.join(PROCESSED_DIR, "features_batters.csv"))
sel26 = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions_2026_bonus.csv"))

lines = []
lines.append("# 2015-2025 Batter Winners Mean Stats by Position")
w_bat = bat[(bat["year"] <= 2025) & (bat["is_winner"] == 1)]
pos_summary = w_bat.groupby("position")[["打率", "本塁打", "打点", "安打", "出塁率", "長打率", "OPS", "RC", "RC27", "pos_starts", "規定打席到達", "is_champion"]].mean().round(3)
lines.append(pos_summary.to_string())

lines.append("\n# 2015-2025 Pitcher Winners Mean Stats")
w_pit = pit[(pit["year"] <= 2025) & (pit["is_winner"] == 1)]
pit_summary = w_pit[["防御率", "勝利", "敗北", "勝率", "投球回", "奪三振", "WHIP", "DIPS", "規定投球回到達", "is_champion", "主要タイトル数_先発"]].mean().round(3)
lines.append(pit_summary.to_string())

lines.append("\n# 2026 Predicted Winners Stats")
for _, r in sel26.iterrows():
    if r["position"] == "投手":
        m = pit[(pit["year"] == 2026) & (pit["league"] == r["league"]) & (pit["norm_name"] == r["norm_name"])].iloc[0]
        lines.append(f"{r['league']} | {r['position']} | {r['選手名']} | {r['チーム']} ({int(m['team_rank'])}位) | 防御率 {m['防御率']:.2f}, {int(m['勝利'])}勝{int(m['敗北'])}敗, 勝率 {m['勝率']:.3f}, 投球回 {m['投球回']:.1f}, 奪三振 {int(m['奪三振'])}, WHIP {m['WHIP']:.2f} | Score={r['pred_score']:.3f}")
    else:
        m = bat[(bat["year"] == 2026) & (bat["league"] == r["league"]) & (bat["position"] == r["position"]) & (bat["norm_name"] == r["norm_name"])].iloc[0]
        lines.append(f"{r['league']} | {r['position']} | {r['選手名']} | {r['チーム']} ({int(m['team_rank'])}位) | 打率 {m['打率']:.3f}, {int(m['本塁打'])}本, {int(m['打点'])}打点, {int(m['安打'])}安打, OPS {m['OPS']:.3f}, RC {m['RC']:.1f}, RC27 {m['RC27']:.2f}, 守備スタメン {int(m['pos_starts'])}試合 | Score={r['pred_score']:.3f}")

with open(os.path.join(OUTPUT_DIR, "summary_tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

for img in ["eda_bestnine_analysis.png", "feature_importance.png", "evaluation_results.png"]:
    src = os.path.join(OUTPUT_DIR, img)
    dst = os.path.join(ARTIFACT_DIR, img)
    if os.path.exists(src):
        shutil.copy2(src, dst)

print("Summary tables written and charts copied!")
