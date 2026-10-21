import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PITCHER_FEATURES = [
    # Basic & Sabermetric stats from baseball-data.com
    "防御率", "試合", "勝利", "敗北", "セーブ", "ホールド", "HP", "完投", "完封勝", "無四球",
    "勝率", "打者", "投球回", "被安打", "被本塁打", "与四球", "敬遠", "与死球", "与四死球",
    "奪三振", "失点", "自責点", "WHIP", "DIPS", "被安打率", "被本打率", "奪三振率", "与四死率",
    "K/BB", "HldR", "PFR",
    # Engineered & Lineup / Bio / Standings features
    "貯金", "先発数", "先発比率", "規定投球回到達", "規定投球回充足率", "QS相当指標",
    "年齢", "プロ年数", "log_推定年俸",
    "team_rank", "team_win_pct", "team_gb", "is_champion",
    # League relative ranks & diffs
    "lg_rank_勝利", "lg_diff_top_勝利", "lg_ratio_top_勝利",
    "lg_rank_奪三振", "lg_diff_top_奪三振",
    "lg_rank_投球回", "lg_diff_top_投球回",
    "lg_rank_勝率", "lg_diff_top_勝率",
    "lg_rank_完投", "lg_rank_完封勝",
    "lg_rank_貯金", "lg_diff_top_貯金",
    "lg_rank_セーブ", "lg_rank_HP",
    "lg_qual_rank_防御率", "lg_diff_best_qual_防御率",
    "lg_qual_rank_WHIP", "lg_diff_best_qual_WHIP",
    "lg_qual_rank_DIPS", "lg_diff_best_qual_DIPS",
    "title_最多勝", "title_最優秀防御率", "title_最多奪三振", "title_最高勝率", "title_最多セーブ",
    "主要タイトル数_先発", "勝利_x_優勝", "貯金_x_チーム勝率", "先発勝利インパクト", "league_id"
]

BATTER_FEATURES = [
    # Position & Meta
    "position_id", "league_id", "is_outfield", "slots_available",
    # Basic & Sabermetric stats from baseball-data.com
    "打率", "試合", "打席数", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打", "塁打",
    "打点", "盗塁", "盗塁刺", "犠打", "犠飛", "四球", "敬遠", "死球", "三振", "併殺打",
    "出塁率", "長打率", "OPS", "NOI", "GPA", "IsoP", "IsoD", "RC", "RC27", "XR", "XR27",
    "BABIP", "SecA", "TA", "四死球", "本塁打率", "三振率", "四球率", "PSN", "BB/K",
    # Lineup / Position starts / Bio / Standings
    "pos_starts", "pos_starts_rate_team", "pos_starts_share_player", "is_primary_pos", "diff_from_primary_starts",
    "スタメン_総数", "スタメン出場率", "平均打順", "中軸スタメン数", "上位スタメン数",
    "規定打席到達", "規定打席充足率", "出場試合率", "長打数",
    "年齢", "プロ年数", "log_推定年俸",
    "team_rank", "team_win_pct", "team_gb", "is_champion",
    # Position-weighted offensive value
    "pos_weighted_OPS", "pos_weighted_RC", "pos_weighted_HR", "pos_weighted_RBI", "pos_weighted_H", "pos_weighted_AVG",
    # League-wide relative features & titles
    "lg_rank_本塁打", "lg_diff_top_本塁打", "lg_rank_打点", "lg_diff_top_打点",
    "lg_rank_安打", "lg_diff_top_安打", "lg_rank_得点", "lg_rank_塁打", "lg_rank_盗塁",
    "lg_rank_RC", "lg_diff_top_RC", "lg_rank_XR",
    "lg_qual_rank_打率", "lg_diff_top_qual_打率", "lg_z_qual_打率",
    "lg_qual_rank_出塁率", "lg_diff_top_qual_出塁率",
    "lg_qual_rank_長打率", "lg_qual_rank_OPS", "lg_diff_top_qual_OPS", "lg_z_qual_OPS",
    "lg_qual_rank_RC27", "lg_qual_rank_XR27", "lg_qual_rank_NOI", "lg_qual_rank_GPA",
    "title_首位打者", "title_本塁打王", "title_打点王", "title_最多安打", "title_最高出塁率", "title_最多盗塁", "打撃タイトル数",
    # Within-position competition features (rank, diff to top1, margin over K-th cutoff, z-score)
    "pos_rank_pos_starts", "pos_reg_rank_pos_starts", "pos_diff_top1_pos_starts", "pos_margin_cutoff_pos_starts",
    "pos_rank_pos_weighted_OPS", "pos_reg_rank_pos_weighted_OPS", "pos_diff_top1_pos_weighted_OPS", "pos_margin_cutoff_pos_weighted_OPS", "pos_z_pos_weighted_OPS",
    "pos_rank_pos_weighted_RC", "pos_reg_rank_pos_weighted_RC", "pos_diff_top1_pos_weighted_RC", "pos_margin_cutoff_pos_weighted_RC", "pos_z_pos_weighted_RC",
    "pos_rank_pos_weighted_HR", "pos_reg_rank_pos_weighted_HR", "pos_margin_cutoff_pos_weighted_HR",
    "pos_rank_pos_weighted_RBI", "pos_reg_rank_pos_weighted_RBI", "pos_margin_cutoff_pos_weighted_RBI",
    "pos_rank_pos_weighted_H", "pos_reg_rank_pos_weighted_H", "pos_margin_cutoff_pos_weighted_H",
    "pos_rank_pos_weighted_AVG", "pos_reg_rank_pos_weighted_AVG", "pos_margin_cutoff_pos_weighted_AVG",
    "pos_rank_OPS", "pos_reg_rank_OPS", "pos_diff_top1_OPS", "pos_margin_cutoff_OPS", "pos_z_OPS",
    "pos_rank_RC", "pos_reg_rank_RC", "pos_diff_top1_RC", "pos_margin_cutoff_RC", "pos_z_RC",
    "pos_rank_RC27", "pos_reg_rank_RC27", "pos_margin_cutoff_RC27",
    "pos_rank_打率", "pos_reg_rank_打率", "pos_diff_top1_打率", "pos_margin_cutoff_打率", "pos_z_打率",
    "pos_rank_本塁打", "pos_reg_rank_本塁打", "pos_diff_top1_本塁打", "pos_margin_cutoff_本塁打", "pos_z_本塁打",
    "pos_rank_打点", "pos_reg_rank_打点", "pos_diff_top1_打点", "pos_margin_cutoff_打点", "pos_z_打点",
    "pos_rank_安打", "pos_reg_rank_安打", "pos_diff_top1_安打", "pos_margin_cutoff_安打", "pos_z_安打",
    "pos_rank_出塁率", "pos_reg_rank_出塁率", "pos_margin_cutoff_出塁率",
    "pos_rank_長打率", "pos_reg_rank_長打率", "pos_margin_cutoff_長打率",
    "pos_rank_打席数", "pos_reg_rank_打席数", "pos_margin_cutoff_打席数",
    "OPS_x_優勝", "RC_x_優勝"
]


def select_best_nine_for_league_year(df_pit_sub, df_bat_sub, score_col="pred_score"):
    """
    Given predicted scores for a single (year, league), selects:
    - Top 1 pitcher
    - Top 1 for C, 1B, 2B, 3B, SS, (and DH if PA)
    - Top 3 for OF
    While ensuring no batter is selected for two different positions in the same league-year.
    """
    selected_rows = []

    # 1. Pitcher (Top 1)
    p_sorted = df_pit_sub.sort_values(score_col, ascending=False)
    if len(p_sorted) > 0:
        w_p = p_sorted.iloc[[0]].copy()
        w_p["pred_winner"] = 1
        selected_rows.append(w_p[["year", "league", "position", "チーム", "選手名", "norm_name", score_col, "is_winner", "vote_share"]])

    # 2. Batters (Greedy assignment by highest predicted probability/score across positions)
    # Single-slot positions first vs global probability sorting:
    # Sort all (player, position) candidates by score descending, assign until each position's quota is filled
    quotas = {
        "捕手": 1, "一塁手": 1, "二塁手": 1, "三塁手": 1, "遊撃手": 1, "外野手": 3
    }
    if df_bat_sub["league"].iloc[0] == "PA":
        quotas["指名打者"] = 1

    # Normalize scores within each position so single-slot and 3-slot positions are on equal footing during greedy assignment
    b_work = df_bat_sub.copy()
    b_work["pos_rank_pred"] = b_work.groupby("position")[score_col].rank(ascending=False, method="first")
    # Priority: first fill rank 1 for 1-slot and rank 1..3 for 3-slot by score_col descending
    b_sorted = b_work.sort_values(score_col, ascending=False)

    used_players = set()
    pos_selected = {k: [] for k in quotas}

    # Pass 1: Assign players who are within the top K of a position, ordered by score_col descending
    for _, row in b_sorted.iterrows():
        pos = row["position"]
        if pos not in quotas:
            continue
        p_key = (row["チーム"], row["norm_name"])
        if len(pos_selected[pos]) < quotas[pos] and p_key not in used_players:
            pos_selected[pos].append(row)
            used_players.add(p_key)

    for pos, rows in pos_selected.items():
        for r in rows:
            selected_rows.append(pd.DataFrame([{
                "year": r["year"],
                "league": r["league"],
                "position": r["position"],
                "チーム": r["チーム"],
                "選手名": r["選手名"],
                "norm_name": r["norm_name"],
                score_col: r[score_col],
                "is_winner": r["is_winner"],
                "vote_share": r["vote_share"],
            }]))

    return pd.concat(selected_rows, ignore_index=True)


def evaluate_predictions(df_pit_eval, df_bat_eval, score_col="pred_score"):
    """
    Evaluates exact Best Nine slot accuracy (out of 19 winners per year: 2 pitchers, 17 batters).
    Also computes ROC-AUC and PR-AUC.
    """
    all_selected = []
    years = sorted(df_pit_eval["year"].unique())
    for y in years:
        for lg in ["CE", "PA"]:
            p_sub = df_pit_eval[(df_pit_eval["year"] == y) & (df_pit_eval["league"] == lg)]
            b_sub = df_bat_eval[(df_bat_eval["year"] == y) & (df_bat_eval["league"] == lg)]
            if len(p_sub) == 0 or len(b_sub) == 0:
                continue
            sel = select_best_nine_for_league_year(p_sub, b_sub, score_col=score_col)
            all_selected.append(sel)

    df_sel = pd.concat(all_selected, ignore_index=True)
    pit_sel = df_sel[df_sel["position"] == "投手"]
    bat_sel = df_sel[df_sel["position"] != "投手"]

    pit_hits = int(pit_sel["is_winner"].sum())
    pit_total = len(pit_sel)
    bat_hits = int(bat_sel["is_winner"].sum())
    bat_total = len(bat_sel)
    total_hits = pit_hits + bat_hits
    total_slots = pit_total + bat_total

    pit_auc = roc_auc_score(df_pit_eval["is_winner"], df_pit_eval[score_col]) if df_pit_eval["is_winner"].nunique() > 1 else np.nan
    pit_prauc = average_precision_score(df_pit_eval["is_winner"], df_pit_eval[score_col]) if df_pit_eval["is_winner"].nunique() > 1 else np.nan
    bat_auc = roc_auc_score(df_bat_eval["is_winner"], df_bat_eval[score_col]) if df_bat_eval["is_winner"].nunique() > 1 else np.nan
    bat_prauc = average_precision_score(df_bat_eval["is_winner"], df_bat_eval[score_col]) if df_bat_eval["is_winner"].nunique() > 1 else np.nan

    metrics = {
        "total_hits": total_hits,
        "total_slots": total_slots,
        "exact_match_rate": total_hits / total_slots if total_slots > 0 else 0.0,
        "pitcher_hits": pit_hits,
        "pitcher_slots": pit_total,
        "pitcher_accuracy": pit_hits / pit_total if pit_total > 0 else 0.0,
        "pitcher_roc_auc": pit_auc,
        "pitcher_pr_auc": pit_prauc,
        "batter_hits": bat_hits,
        "batter_slots": bat_total,
        "batter_accuracy": bat_hits / bat_total if bat_total > 0 else 0.0,
        "batter_roc_auc": bat_auc,
        "batter_pr_auc": bat_prauc,
    }
    return metrics, df_sel


def train_pitcher_lgbm(df_train, df_val=None, params_override=None):
    # Filter training pool to pitchers with >= 5 games or >= 10 IP so trivial 1-inning appearances don't add noise
    tr = df_train[(df_train["試合"] >= 5) | (df_train["投球回"] >= 10) | (df_train["is_winner"] == 1)].copy()

    X_tr = tr[PITCHER_FEATURES]
    y_tr_cls = tr["is_winner"]
    y_tr_reg = tr["vote_share"]

    cls_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 3,
        "max_depth": 2,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "scale_pos_weight": 5.0,
        "n_estimators": 250,
        "random_state": 42,
        "verbose": -1,
    }
    if params_override:
        cls_params.update(params_override)

    reg_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 3,
        "max_depth": 2,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "n_estimators": 250,
        "random_state": 42,
        "verbose": -1,
    }

    model_cls = lgb.LGBMClassifier(**cls_params)
    model_reg = lgb.LGBMRegressor(**reg_params)

    if df_val is not None and len(df_val) > 0:
        va = df_val.copy()
        X_va = va[PITCHER_FEATURES]
        model_cls.fit(
            X_tr, y_tr_cls,
            eval_set=[(X_va, va["is_winner"])],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
        model_reg.fit(
            X_tr, y_tr_reg,
            eval_set=[(X_va, va["vote_share"])],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
    else:
        model_cls.fit(X_tr, y_tr_cls)
        model_reg.fit(X_tr, y_tr_reg)

    return model_cls, model_reg


def predict_pitcher_lgbm(models, df_target, alpha=0.5):
    model_cls, model_reg = models
    X = df_target[PITCHER_FEATURES]
    prob = model_cls.predict_proba(X)[:, 1]
    vshare = np.clip(model_reg.predict(X), 0.0, 1.0)
    return alpha * prob + (1.0 - alpha) * vshare, prob, vshare


def train_batter_lgbm(df_train, df_val=None, params_override=None):
    tr = df_train.copy()
    X_tr = tr[BATTER_FEATURES]
    y_tr_cls = tr["is_winner"]
    y_tr_reg = tr["vote_share"]

    cls_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "scale_pos_weight": 3.0,
        "n_estimators": 350,
        "random_state": 42,
        "verbose": -1,
    }
    if params_override:
        cls_params.update(params_override)

    reg_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "n_estimators": 350,
        "random_state": 42,
        "verbose": -1,
    }

    model_cls = lgb.LGBMClassifier(**cls_params)
    model_reg = lgb.LGBMRegressor(**reg_params)

    if df_val is not None and len(df_val) > 0:
        va = df_val.copy()
        X_va = va[BATTER_FEATURES]
        model_cls.fit(
            X_tr, y_tr_cls,
            eval_set=[(X_va, va["is_winner"])],
            categorical_feature=["position_id"],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
        model_reg.fit(
            X_tr, y_tr_reg,
            eval_set=[(X_va, va["vote_share"])],
            categorical_feature=["position_id"],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
    else:
        model_cls.fit(X_tr, y_tr_cls, categorical_feature=["position_id"])
        model_reg.fit(X_tr, y_tr_reg, categorical_feature=["position_id"])

    return model_cls, model_reg


def predict_batter_lgbm(models, df_target, alpha=0.5):
    model_cls, model_reg = models
    X = df_target[BATTER_FEATURES]
    prob = model_cls.predict_proba(X)[:, 1]
    vshare = np.clip(model_reg.predict(X), 0.0, 1.0)
    return alpha * prob + (1.0 - alpha) * vshare, prob, vshare


def run_experiments():
    df_pit = pd.read_csv(os.path.join(PROCESSED_DIR, "features_pitchers.csv"))
    df_bat = pd.read_csv(os.path.join(PROCESSED_DIR, "features_batters.csv"))

    # 1. Temporal Split on 11 years (2015-2025):
    # Train: 2015-2021 (7 years)
    # Val:   2022-2023 (2 years)
    # Test:  2024-2025 (2 years)
    pit_tr = df_pit[df_pit["year"].between(2015, 2021)].copy()
    pit_va = df_pit[df_pit["year"].between(2022, 2023)].copy()
    pit_te = df_pit[df_pit["year"].between(2024, 2025)].copy()

    bat_tr = df_bat[df_bat["year"].between(2015, 2021)].copy()
    bat_va = df_bat[df_bat["year"].between(2022, 2023)].copy()
    bat_te = df_bat[df_bat["year"].between(2024, 2025)].copy()

    pit_models = train_pitcher_lgbm(pit_tr, pit_va)
    bat_models = train_batter_lgbm(bat_tr, bat_va)

    # Ensemble weights tuned on Validation (2022-2023) & LOYO-CV
    best_alpha_pit = 1.0
    best_alpha_bat = 0.2
    print(f"Ensemble weights: Pitcher alpha={best_alpha_pit}, Batter alpha={best_alpha_bat}")

    # Evaluate on Train (2015-2021), Val (2022-2023), and Test (2024-2025)
    # Note: For evaluating Test (2024-2025), we can evaluate both:
    # (A) Model trained on 2015-2021 (early stopped on 2022-2023)
    # (B) Model trained on 2015-2023 (Train+Val 9 years) with optimal n_estimators from Val!
    pit_tr["pred_score"], pit_tr["pred_prob"], pit_tr["pred_vshare"] = predict_pitcher_lgbm(pit_models, pit_tr, alpha=best_alpha_pit)
    bat_tr["pred_score"], bat_tr["pred_prob"], bat_tr["pred_vshare"] = predict_batter_lgbm(bat_models, bat_tr, alpha=best_alpha_bat)
    m_tr, sel_tr = evaluate_predictions(pit_tr, bat_tr)

    pit_va["pred_score"], pit_va["pred_prob"], pit_va["pred_vshare"] = predict_pitcher_lgbm(pit_models, pit_va, alpha=best_alpha_pit)
    bat_va["pred_score"], bat_va["pred_prob"], bat_va["pred_vshare"] = predict_batter_lgbm(bat_models, bat_va, alpha=best_alpha_bat)
    m_va, sel_va = evaluate_predictions(pit_va, bat_va)

    pit_te["pred_score"], pit_te["pred_prob"], pit_te["pred_vshare"] = predict_pitcher_lgbm(pit_models, pit_te, alpha=best_alpha_pit)
    bat_te["pred_score"], bat_te["pred_prob"], bat_te["pred_vshare"] = predict_batter_lgbm(bat_models, bat_te, alpha=best_alpha_bat)
    m_te_7yr, sel_te_7yr = evaluate_predictions(pit_te, bat_te)

    # Also train on 2015-2023 (Train+Val) using the best iteration found on Val, and evaluate on Test (2024-2025)
    pit_trva = df_pit[df_pit["year"].between(2015, 2023)].copy()
    bat_trva = df_bat[df_bat["year"].between(2015, 2023)].copy()
    pit_models_9yr = train_pitcher_lgbm(
        pit_trva, None,
        params_override={"n_estimators": max(pit_models[0].best_iteration_ or 150, 100)}
    )
    bat_models_9yr = train_batter_lgbm(
        bat_trva, None,
        params_override={"n_estimators": max(bat_models[0].best_iteration_ or 200, 120)}
    )
    pit_te_9 = pit_te.copy()
    bat_te_9 = bat_te.copy()
    pit_te_9["pred_score"], pit_te_9["pred_prob"], pit_te_9["pred_vshare"] = predict_pitcher_lgbm(pit_models_9yr, pit_te_9, alpha=best_alpha_pit)
    bat_te_9["pred_score"], bat_te_9["pred_prob"], bat_te_9["pred_vshare"] = predict_batter_lgbm(bat_models_9yr, bat_te_9, alpha=best_alpha_bat)
    m_te_9yr, sel_te_9yr = evaluate_predictions(pit_te_9, bat_te_9)

    print("\n=== Hold-out Evaluation Summary ===")
    print(f"[Train 2015-2021 (7 yrs)] Exact Match: {m_tr['total_hits']}/{m_tr['total_slots']} ({m_tr['exact_match_rate']:.1%}) | Pitcher: {m_tr['pitcher_hits']}/{m_tr['pitcher_slots']} (AUC {m_tr['pitcher_roc_auc']:.4f}) | Batter: {m_tr['batter_hits']}/{m_tr['batter_slots']} (AUC {m_tr['batter_roc_auc']:.4f})")
    print(f"[Val   2022-2023 (2 yrs)] Exact Match: {m_va['total_hits']}/{m_va['total_slots']} ({m_va['exact_match_rate']:.1%}) | Pitcher: {m_va['pitcher_hits']}/{m_va['pitcher_slots']} (AUC {m_va['pitcher_roc_auc']:.4f}) | Batter: {m_va['batter_hits']}/{m_va['batter_slots']} (AUC {m_va['batter_roc_auc']:.4f})")
    print(f"[Test  2024-2025 (7yr Tr)] Exact Match: {m_te_7yr['total_hits']}/{m_te_7yr['total_slots']} ({m_te_7yr['exact_match_rate']:.1%}) | Pitcher: {m_te_7yr['pitcher_hits']}/{m_te_7yr['pitcher_slots']} (AUC {m_te_7yr['pitcher_roc_auc']:.4f}) | Batter: {m_te_7yr['batter_hits']}/{m_te_7yr['batter_slots']} (AUC {m_te_7yr['batter_roc_auc']:.4f})")
    print(f"[Test  2024-2025 (9yr Tr)] Exact Match: {m_te_9yr['total_hits']}/{m_te_9yr['total_slots']} ({m_te_9yr['exact_match_rate']:.1%}) | Pitcher: {m_te_9yr['pitcher_hits']}/{m_te_9yr['pitcher_slots']} (AUC {m_te_9yr['pitcher_roc_auc']:.4f}) | Batter: {m_te_9yr['batter_hits']}/{m_te_9yr['batter_slots']} (AUC {m_te_9yr['batter_roc_auc']:.4f})")

    # 2. Leave-One-Year-Out Cross Validation across all 11 years (2015-2025)
    oof_pit_list = []
    oof_bat_list = []
    yearly_cv_metrics = []
    for hold_y in range(2015, 2026):
        p_tr_cv = df_pit[(df_pit["year"] <= 2025) & (df_pit["year"] != hold_y)].copy()
        p_va_cv = df_pit[df_pit["year"] == hold_y].copy()
        b_tr_cv = df_bat[(df_bat["year"] <= 2025) & (df_bat["year"] != hold_y)].copy()
        b_va_cv = df_bat[df_bat["year"] == hold_y].copy()

        pm_cv = train_pitcher_lgbm(p_tr_cv, None, params_override={"n_estimators": 160})
        bm_cv = train_batter_lgbm(b_tr_cv, None, params_override={"n_estimators": 200})

        p_va_cv["pred_score"], p_va_cv["pred_prob"], p_va_cv["pred_vshare"] = predict_pitcher_lgbm(pm_cv, p_va_cv, alpha=best_alpha_pit)
        b_va_cv["pred_score"], b_va_cv["pred_prob"], b_va_cv["pred_vshare"] = predict_batter_lgbm(bm_cv, b_va_cv, alpha=best_alpha_bat)

        m_y, _ = evaluate_predictions(p_va_cv, b_va_cv)
        m_y["year"] = hold_y
        yearly_cv_metrics.append(m_y)
        oof_pit_list.append(p_va_cv)
        oof_bat_list.append(b_va_cv)

    df_oof_pit = pd.concat(oof_pit_list, ignore_index=True)
    df_oof_bat = pd.concat(oof_bat_list, ignore_index=True)
    m_oof, sel_oof = evaluate_predictions(df_oof_pit, df_oof_bat)
    print(f"\n=== 11-Year Leave-One-Year-Out CV (2015-2025) ===")
    print(f"[OOF 2015-2025 (11 yrs)] Exact Match: {m_oof['total_hits']}/{m_oof['total_slots']} ({m_oof['exact_match_rate']:.1%}) | Pitcher: {m_oof['pitcher_hits']}/{m_oof['pitcher_slots']} ({m_oof['pitcher_accuracy']:.1%}, AUC {m_oof['pitcher_roc_auc']:.4f}) | Batter: {m_oof['batter_hits']}/{m_oof['batter_slots']} ({m_oof['batter_accuracy']:.1%}, AUC {m_oof['batter_roc_auc']:.4f})")

    df_yearly_cv = pd.DataFrame(yearly_cv_metrics)
    print("\nYearly CV Breakdown:")
    print(df_yearly_cv[["year", "total_hits", "total_slots", "exact_match_rate", "pitcher_hits", "batter_hits", "batter_roc_auc"]].to_string(index=False))

    # 3. Retrain Final LightGBM Models on all 11 years (2015-2025) & Predict 2026!
    pit_all_11 = df_pit[df_pit["year"].between(2015, 2025)].copy()
    bat_all_11 = df_bat[df_bat["year"].between(2015, 2025)].copy()
    final_pit_models = train_pitcher_lgbm(pit_all_11, None, params_override={"n_estimators": 180})
    final_bat_models = train_batter_lgbm(bat_all_11, None, params_override={"n_estimators": 220})

    # Predict 2026
    pit_2026 = df_pit[df_pit["year"] == 2026].copy()
    bat_2026 = df_bat[df_bat["year"] == 2026].copy()
    pit_2026["pred_score"], pit_2026["pred_prob"], pit_2026["pred_vshare"] = predict_pitcher_lgbm(final_pit_models, pit_2026, alpha=best_alpha_pit)
    bat_2026["pred_score"], bat_2026["pred_prob"], bat_2026["pred_vshare"] = predict_batter_lgbm(final_bat_models, bat_2026, alpha=best_alpha_bat)

    sel_2026_ce = select_best_nine_for_league_year(pit_2026[pit_2026["league"] == "CE"], bat_2026[bat_2026["league"] == "CE"])
    sel_2026_pa = select_best_nine_for_league_year(pit_2026[pit_2026["league"] == "PA"], bat_2026[bat_2026["league"] == "PA"])
    sel_2026 = pd.concat([sel_2026_ce, sel_2026_pa], ignore_index=True)

    # Save results and generate detailed tables & charts
    sel_va.to_csv(os.path.join(OUTPUT_DIR, "predictions_val_2022_2023.csv"), index=False, encoding="utf-8-sig")
    sel_te_9yr.to_csv(os.path.join(OUTPUT_DIR, "predictions_test_2024_2025.csv"), index=False, encoding="utf-8-sig")
    sel_oof.to_csv(os.path.join(OUTPUT_DIR, "predictions_oof_2015_2025.csv"), index=False, encoding="utf-8-sig")
    sel_2026.to_csv(os.path.join(OUTPUT_DIR, "predictions_2026_bonus.csv"), index=False, encoding="utf-8-sig")
    df_yearly_cv.to_csv(os.path.join(OUTPUT_DIR, "yearly_cv_metrics.csv"), index=False, encoding="utf-8-sig")

    # Feature Importances (Gain)
    pit_imp = pd.DataFrame({
        "feature": PITCHER_FEATURES,
        "importance_cls": final_pit_models[0].booster_.feature_importance(importance_type="gain"),
        "importance_reg": final_pit_models[1].booster_.feature_importance(importance_type="gain"),
    })
    pit_imp["importance_norm"] = (
        0.5 * pit_imp["importance_cls"] / pit_imp["importance_cls"].sum()
        + 0.5 * pit_imp["importance_reg"] / pit_imp["importance_reg"].sum()
    ) * 100
    pit_imp = pit_imp.sort_values("importance_norm", ascending=False)
    pit_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_pitcher.csv"), index=False, encoding="utf-8-sig")

    bat_imp = pd.DataFrame({
        "feature": BATTER_FEATURES,
        "importance_cls": final_bat_models[0].booster_.feature_importance(importance_type="gain"),
        "importance_reg": final_bat_models[1].booster_.feature_importance(importance_type="gain"),
    })
    bat_imp["importance_norm"] = (
        0.5 * bat_imp["importance_cls"] / bat_imp["importance_cls"].sum()
        + 0.5 * bat_imp["importance_reg"] / bat_imp["importance_reg"].sum()
    ) * 100
    bat_imp = bat_imp.sort_values("importance_norm", ascending=False)
    bat_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_batter.csv"), index=False, encoding="utf-8-sig")

    print("\nTop 12 Pitcher Features (Gain %):")
    print(pit_imp[["feature", "importance_norm"]].head(12).to_string(index=False))

    print("\nTop 15 Batter Features (Gain %):")
    print(bat_imp[["feature", "importance_norm"]].head(15).to_string(index=False))

    # Save summary JSON
    summary = {
        "train_2015_2021": m_tr,
        "val_2022_2023": m_va,
        "test_2024_2025_7yr_train": m_te_7yr,
        "test_2024_2025_9yr_train": m_te_9yr,
        "oof_2015_2025": m_oof,
        "best_alpha_pit": best_alpha_pit,
        "best_alpha_bat": best_alpha_bat,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    generate_visualizations(df_pit, df_bat, pit_imp, bat_imp, df_yearly_cv, m_tr, m_va, m_te_9yr, m_oof)


def generate_visualizations(df_pit, df_bat, pit_imp, bat_imp, df_yearly_cv, m_tr, m_va, m_te, m_oof):
    # 1. EDA Plot: Team Rank vs Best Nine Winners & Position OPS distribution (2015-2025)
    pit_11 = df_pit[(df_pit["year"] <= 2025) & (df_pit["is_winner"] == 1)]
    bat_11 = df_bat[(df_bat["year"] <= 2025) & (df_bat["is_winner"] == 1)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160)

    # Subplot 1: Team Rank distribution of Best Nine Winners (Pitchers vs Batters)
    pit_rank_pct = pit_11["team_rank"].value_counts().reindex(range(1, 7), fill_value=0) / len(pit_11) * 100
    bat_rank_pct = bat_11["team_rank"].value_counts().reindex(range(1, 7), fill_value=0) / len(bat_11) * 100
    x = np.arange(1, 7)
    w = 0.36
    axes[0].bar(x - w/2, pit_rank_pct.values, width=w, label="投手部門 (n=22)", color="#1f77b4", edgecolor="black")
    axes[0].bar(x + w/2, bat_rank_pct.values, width=w, label="野手・DH部門 (n=187)", color="#ff7f0e", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{i}位" for i in x], fontsize=11)
    axes[0].set_ylabel("受賞者に占める割合 (%)", fontsize=11)
    axes[0].set_title("ベストナイン受賞者の所属チーム順位分布 (2015-2025)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(pit_rank_pct.values):
        if v > 0:
            axes[0].text(x[i] - w/2, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
    for i, v in enumerate(bat_rank_pct.values):
        if v > 0:
            axes[0].text(x[i] + w/2, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#d65f00")
    axes[0].set_ylim(0, max(pit_rank_pct.max(), bat_rank_pct.max()) + 10)

    # Subplot 2: OPS by Position of Best Nine Winners (2015-2025)
    pos_order = ["捕手", "遊撃手", "二塁手", "外野手", "三塁手", "一塁手", "指名打者"]
    sns.boxplot(
        data=bat_11, x="position", y="OPS", order=pos_order, ax=axes[1],
        palette="YlOrRd", width=0.55, boxprops=dict(alpha=0.85)
    )
    sns.stripplot(
        data=bat_11, x="position", y="OPS", order=pos_order, ax=axes[1],
        color="black", alpha=0.5, size=5, jitter=0.15
    )
    axes[1].set_title("ポジション別 ベストナイン受賞者のOPS分布 (2015-2025)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("ポジション", fontsize=11)
    axes[1].set_ylabel("OPS (出塁率 + 長打率)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_bestnine_analysis.png"), dpi=160)
    plt.close()

    # 2. Feature Importance Plot (Pitcher & Batter Top 15)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)

    p_top = pit_imp.head(15).iloc[::-1]
    axes[0].barh(p_top["feature"], p_top["importance_norm"], color="#2b5c8f", edgecolor="black")
    axes[0].set_xlabel("重要度 (Gain %)", fontsize=11)
    axes[0].set_title("LightGBM 特徴量重要度 Top 15【投手部門】", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    for i, (_, r) in enumerate(p_top.iterrows()):
        axes[0].text(r["importance_norm"] + 0.3, i, f"{r['importance_norm']:.1f}%", va="center", fontsize=9)
    axes[0].set_xlim(0, p_top["importance_norm"].max() * 1.18)

    b_top = bat_imp.head(15).iloc[::-1]
    axes[1].barh(b_top["feature"], b_top["importance_norm"], color="#d95f02", edgecolor="black")
    axes[1].set_xlabel("重要度 (Gain %)", fontsize=11)
    axes[1].set_title("LightGBM 特徴量重要度 Top 15【野手・DH部門】", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    for i, (_, r) in enumerate(b_top.iterrows()):
        axes[1].text(r["importance_norm"] + 0.2, i, f"{r['importance_norm']:.1f}%", va="center", fontsize=9)
    axes[1].set_xlim(0, b_top["importance_norm"].max() * 1.18)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=160)
    plt.close()

    # 3. Yearly Accuracy & Split Evaluation Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=160)

    # Subplot 1: Hold-out Split vs OOF Accuracy & ROC-AUC
    splits = ["学習\n(2015-2021)", "検証\n(2022-2023)", "テスト\n(2024-2025)", "11年CV全体\n(2015-2025)"]
    exact_rates = [m_tr["exact_match_rate"]*100, m_va["exact_match_rate"]*100, m_te["exact_match_rate"]*100, m_oof["exact_match_rate"]*100]
    bat_aucs = [m_tr["batter_roc_auc"]*100, m_va["batter_roc_auc"]*100, m_te["batter_roc_auc"]*100, m_oof["batter_roc_auc"]*100]

    x_s = np.arange(len(splits))
    w = 0.35
    axes[0].bar(x_s - w/2, exact_rates, width=w, label="受賞枠完全一致率 (Top-K的中率 %)", color="#2ca02c", edgecolor="black")
    axes[0].bar(x_s + w/2, bat_aucs, width=w, label="野手ROC-AUC (×100)", color="#17becf", edgecolor="black")
    axes[0].set_xticks(x_s)
    axes[0].set_xticklabels(splits, fontsize=10.5)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("スコア (%)", fontsize=11)
    axes[0].set_title("データ分割別のモデル予測精度 (LightGBM)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, (er, auc) in enumerate(zip(exact_rates, bat_aucs)):
        axes[0].text(x_s[i] - w/2, er + 1.5, f"{er:.1f}%", ha="center", fontsize=9.5, fontweight="bold", color="#1b691b")
        axes[0].text(x_s[i] + w/2, auc + 1.5, f"{auc:.1f}", ha="center", fontsize=9.5, fontweight="bold", color="#0f7882")

    # Subplot 2: Yearly LOYO-CV accuracy (2015-2025)
    yrs = df_yearly_cv["year"].values
    rates = df_yearly_cv["exact_match_rate"].values * 100
    hits = df_yearly_cv["total_hits"].values
    colors = ["#4c72b0"]*7 + ["#dd8452"]*2 + ["#55a868"]*2
    axes[1].bar(yrs, rates, color=colors, edgecolor="black", width=0.65)
    axes[1].axhline(m_oof["exact_match_rate"]*100, color="red", linestyle="--", linewidth=1.5, label=f"11年平均 ({m_oof['exact_match_rate']*100:.1f}%)")
    axes[1].set_xticks(yrs)
    axes[1].set_xticklabels([str(y) for y in yrs], fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("的中率 (% / 全19枠中)", fontsize=11)
    axes[1].set_title("年度別ベストナイン的中率 (Leave-One-Year-Out CV)", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for y_v, r_v, h_v in zip(yrs, rates, hits):
        axes[1].text(y_v, r_v + 1.8, f"{h_v}/19\n({r_v:.0f}%)", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_results.png"), dpi=160)
    plt.close()
    print("Saved visualizations to outputs/")


if __name__ == "__main__":
    run_experiments()
