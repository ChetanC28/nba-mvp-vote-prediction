"""Benchmark the adaptive weighted ensemble against the report's model table.

The original driver for this algorithm used its own split (2017-2021 test, 18
features), so its numbers were never comparable to the nine models in the
report. This script runs the same algorithm on the report's setup instead:

    train+val : 1980-2019   (16,906 rows)
    test      : 2020-2021   (1,070 rows)
    features  : the same 24 engineered columns the stacked ensemble used

Writes results/model_comparison.csv and figures/model_comparison.png.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from adaptive_ensemble import AdaptiveWeightedEnsemble

DATA_PATH = Path("cleaned_data.csv")
TARGET_COL = "Pts Won"
SEASON_COL = "Year"
PLAYER_COL = "Player"
TEST_SEASONS_N = 2
VAL_SEASONS_N = 2
RANDOM_STATE = 42

# The nine models already evaluated in the report, for the combined table.
REPORTED_RESULTS = [
    ("Stacked Ensemble", 31.523249, 4.479322, 0.679146),
    ("CatBoost", 31.873950, 4.487180, 0.671967),
    ("LightGBM", 32.256245, 4.458004, 0.664051),
    ("Gradient Boosting", 32.887343, 4.756298, 0.650777),
    ("Random Forest", 34.796803, 5.502821, 0.609047),
    ("XGBoost", 36.136097, 5.007829, 0.578373),
    ("Ridge", 48.705371, 15.481999, 0.234052),
    ("Linear Regression", 49.048371, 16.306498, 0.223226),
    ("Dummy Mean", 55.673798, 11.113900, -0.000800),
]


def engineer_features(dataframe):
    """Same feature engineering as the stacking notebook and app.py."""
    df = dataframe.copy()

    alias_map = {
        "PTS": "PPG", "TRB": "RPG", "AST": "APG", "STL": "SPG",
        "BLK": "BPG", "MP": "MPG", "TOV": "TPG",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    denom = (df["W"] + df["L"]).replace(0, np.nan)
    df["Win_Rate"] = (df["W"] / denom).fillna(0)

    denom = (2 * (df["FGA"] + 0.44 * df["FTA"])).replace(0, np.nan)
    df["TS%"] = (df["PTS"] / denom).fillna(0)

    denom = df["MP"].replace(0, np.nan)
    df["Usage"] = ((df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / denom).fillna(0)

    team_games = (df["W"] + df["L"]).replace(0, np.nan)
    df["Avail_Rate"] = (df["G"] / team_games).fillna(0)

    df["Scoring_Team_Impact"] = df["PPG"] * df["Win_Rate"]
    df["Efficiency_Availability"] = df["TS%"] * df["Avail_Rate"]

    return df


CANDIDATE_FEATURES = [
    "W", "L", "Win_Rate", "PS/G", "PA/G",
    "PPG", "RPG", "APG", "SPG", "BPG", "MPG", "TPG",
    "FG%", "3P%", "2P%", "eFG%", "FT%", "TS%",
    "Usage", "G", "Age", "Avail_Rate",
    "Scoring_Team_Impact", "Efficiency_Availability",
]


def base_model_configs():
    """The five base learners from the original adaptive-ensemble script."""
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=600, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=900, max_depth=7, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=700, max_depth=10, learning_rate=0.07,
            num_leaves=63, min_child_samples=30, subsample=0.85,
            colsample_bytree=0.85, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=700, depth=8, learning_rate=0.08,
            l2_leaf_reg=2.0, random_state=RANDOM_STATE, verbose=False,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=700, max_depth=7, learning_rate=0.07,
            min_samples_split=15, min_samples_leaf=6, subsample=0.85,
            max_features="sqrt", random_state=RANDOM_STATE,
        ),
    }


def get_oof_predictions(X_train, y_train, X_test, train_years, n_splits=5):
    """Contiguous-season-block OOF predictions (as in the original script)."""
    unique_years = sorted(np.unique(train_years))
    fold_size = len(unique_years) // n_splits
    configs = base_model_configs()

    oof_train = np.zeros((len(X_train), len(configs)))
    oof_test = np.zeros((len(X_test), len(configs)))

    for m_idx, (name, base_model) in enumerate(configs.items()):
        print(f"  OOF for {name}...")
        test_preds_folds = []

        for fold in range(n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < n_splits - 1 else len(unique_years)
            val_years = unique_years[start:end]

            val_mask = np.isin(train_years, val_years)
            train_mask = ~val_mask

            fold_model = clone(base_model)
            fold_model.fit(X_train[train_mask], y_train[train_mask])
            oof_train[val_mask, m_idx] = fold_model.predict(X_train[val_mask])
            test_preds_folds.append(fold_model.predict(X_test))

        oof_test[:, m_idx] = np.mean(test_preds_folds, axis=0)

    return oof_train, oof_test


def season_top1_accuracy(eval_df, pred_col):
    scores = []
    for _, grp in eval_df.groupby(SEASON_COL):
        actual_top = grp.sort_values(TARGET_COL, ascending=False).iloc[0][PLAYER_COL]
        pred_top = grp.sort_values(pred_col, ascending=False).iloc[0][PLAYER_COL]
        scores.append(int(actual_top == pred_top))
    return float(np.mean(scores))


def season_topk_overlap(eval_df, pred_col, k=5):
    overlaps = []
    for _, grp in eval_df.groupby(SEASON_COL):
        actual_topk = set(grp.sort_values(TARGET_COL, ascending=False).head(k)[PLAYER_COL])
        pred_topk = set(grp.sort_values(pred_col, ascending=False).head(k)[PLAYER_COL])
        overlaps.append(len(actual_topk & pred_topk) / k)
    return float(np.mean(overlaps))


def diagnose_gating(ensemble, oof_train, oof_test, y_trainval, y_test):
    """Check what the disagreement gate is actually separating.

    The gate thresholds on np.std of the raw base predictions. Those predictions
    are in MVP-vote-point units, so their spread scales with their magnitude —
    which means the threshold may be sorting players by how good they are rather
    than by how much the models genuinely disagree.
    """
    print("\n" + "-" * 78)
    print("GATING DIAGNOSTIC")
    print("-" * 78)

    thr = ensemble.disagreement_threshold
    print(f"Disagreement threshold (median train std): {thr:.3f} vote points")

    for label, oof, y in [("train+val", oof_train, y_trainval), ("test", oof_test, y_test)]:
        dis = np.std(oof, axis=1)
        votes = np.asarray(y) > 0
        above = dis > thr
        print(f"\n{label}:")
        print(f"  rows                                 {len(y)}")
        print(f"  actually received MVP votes          {votes.sum()} ({votes.mean():.1%})")
        print(f"  routed to 'low agreement' branch     {above.sum()} ({above.mean():.1%})")
        if votes.sum():
            print(f"  vote-getters routed 'low agreement'  {above[votes].mean():.1%}")
        if (~votes).sum():
            print(f"  zero-vote routed 'low agreement'     {above[~votes].mean():.1%}")

    # Weight floor: how many base models were pinned at the min_weight bound?
    print("\nLearned weights (RF, XGB, LGBM, CatBoost, GB):")
    for name, w in [
        ("global        ", ensemble.global_weights),
        ("high agreement", ensemble.high_agreement_weights),
        ("low agreement ", ensemble.low_agreement_weights),
    ]:
        pinned = int(np.sum(np.isclose(w, 0.05, atol=1e-3)))
        print(f"  {name}: {np.round(w, 3)}   ({pinned}/5 pinned at the 0.05 floor)")

    # Does the optimiser actually give more weight to the better base learners?
    names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]
    base_rmse = np.array([
        np.sqrt(mean_squared_error(y_test, np.clip(oof_test[:, i], 0, None)))
        for i in range(oof_test.shape[1])
    ])
    w = ensemble.global_weights

    print("\nBase learner quality vs. assigned global weight:")
    print("  (each base learner here is the mean of its 5 fold models)")
    for i in np.argsort(base_rmse):
        print(f"    {names[i]:20s} test RMSE {base_rmse[i]:6.2f}   weight {w[i]:.3f}")

    rho = spearmanr(base_rmse, w).statistic
    print(f"\n  Spearman(test RMSE, weight) = {rho:+.2f}"
          "   [positive => more weight on WORSE models]")
    print(f"  Best single base learner:     {base_rmse.min():.2f} "
          f"({names[int(np.argmin(base_rmse))]})")
    print("-" * 78)


def main():
    df = engineer_features(pd.read_csv(DATA_PATH))
    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]

    all_seasons = sorted(df[SEASON_COL].dropna().unique())
    test_seasons = all_seasons[-TEST_SEASONS_N:]
    trainval_seasons = all_seasons[:-TEST_SEASONS_N]

    trainval_df = df[df[SEASON_COL].isin(trainval_seasons)].copy()
    test_df = df[df[SEASON_COL].isin(test_seasons)].copy()

    train_only = df[df[SEASON_COL].isin(all_seasons[:-(TEST_SEASONS_N + VAL_SEASONS_N)])]
    medians = train_only[feature_cols].median(numeric_only=True)

    X_trainval = trainval_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).values
    y_trainval = trainval_df[TARGET_COL].values
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).values
    y_test = test_df[TARGET_COL].values

    print(f"Features: {len(feature_cols)}")
    print(f"Train+val: {X_trainval.shape}  seasons {trainval_seasons[0]:.0f}-{trainval_seasons[-1]:.0f}")
    print(f"Test:      {X_test.shape}  seasons {[int(s) for s in test_seasons]}\n")

    # The OOF loop is the expensive half (~10 min). Cache it so the analysis
    # below can be re-run cheaply. Delete results/oof_cache.npz to force a refit.
    cache = Path("results/oof_cache.npz")
    if cache.exists():
        print(f"Loading cached OOF predictions from {cache}")
        z = np.load(cache)
        oof_train, oof_test = z["oof_train"], z["oof_test"]
    else:
        print("Generating out-of-fold predictions...")
        oof_train, oof_test = get_oof_predictions(
            X_trainval, y_trainval, X_test, trainval_df[SEASON_COL].values, n_splits=5
        )
        Path("results").mkdir(exist_ok=True)
        np.savez_compressed(cache, oof_train=oof_train, oof_test=oof_test)

    print("\nFitting adaptive weighted ensemble...")
    ensemble = AdaptiveWeightedEnsemble(n_base_models=oof_train.shape[1])
    ensemble.fit(oof_train, trainval_df[SEASON_COL].values, y_trainval)

    y_pred, _ = ensemble.predict(oof_test)
    y_pred = np.clip(y_pred, 0, None)

    y_pred_avg = np.clip(oof_test.mean(axis=1), 0, None)

    rows = []
    for name, pred in [("Adaptive Weighted Ensemble", y_pred), ("Simple Average (5 models)", y_pred_avg)]:
        local = test_df.copy()
        local["pred"] = pred
        rows.append({
            "Model": name,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
            "Top1": season_top1_accuracy(local, "pred"),
            "Top5": season_topk_overlap(local, "pred"),
            "Source": "this script",
        })

    for name, rmse_v, mae_v, r2_v in REPORTED_RESULTS:
        rows.append({
            "Model": name, "RMSE": rmse_v, "MAE": mae_v, "R2": r2_v,
            "Top1": 0.0 if name == "Dummy Mean" else 0.5,
            "Top5": 0.0 if name == "Dummy Mean" else (0.8 if name == "Random Forest" else 0.7),
            "Source": "report",
        })

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    print("\n" + "=" * 78)
    print(results.to_string(index=False))
    print("=" * 78)

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    results.to_csv("results/model_comparison.csv", index=False)

    diagnose_gating(ensemble, oof_train, oof_test, y_trainval, y_test)

    plot_df = results[results["Model"] != "Simple Average (5 models)"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [
        "#1E3A8A" if m == "Stacked Ensemble"
        else "#0891B2" if m == "Adaptive Weighted Ensemble"
        else "#CBD5E1"
        for m in plot_df["Model"]
    ]
    ax.bar(plot_df["Model"], plot_df["RMSE"], color=colors)
    for i, v in enumerate(plot_df["RMSE"]):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8.5, color="#334155")
    ax.set_ylabel("Test RMSE (MVP vote points)")
    ax.set_title("Held-out 2020-21 seasons — lower is better", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/model_comparison.png", dpi=160)
    print("\nWrote results/model_comparison.csv and figures/model_comparison.png")


if __name__ == "__main__":
    main()
