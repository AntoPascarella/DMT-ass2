import os
import numpy as np
import pandas as pd

# ============================================================
# TASK 5 - DEPLOYMENT / ETHICAL AI
# Query-level provider-fairness reranking
# Group variable: prop_brand_bool / propbrandbool
# 0 = independent hotel, 1 = chain hotel
# ============================================================

# -------------------------
# 1) Helpers
# -------------------------
def get_df(*names):
    g = globals()
    for name in names:
        if name in g and isinstance(g[name], pd.DataFrame):
            return g[name]
    raise ValueError(f"Could not find any dataframe among: {names}")

def get_array(*names):
    g = globals()
    for name in names:
        if name in g:
            return np.asarray(g[name], dtype=float), name
    return None, None

def get_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of these columns: {candidates}")

def per_query_zscore(scores, qids):
    s = pd.Series(np.asarray(scores, dtype=float))
    q = pd.Series(qids)
    grp = s.groupby(q)
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, 1).fillna(1)
    return ((s - mean) / std).values

def ndcg_at_5(scores, relevance, query_ids):
    df_eval = pd.DataFrame({
        "score": np.asarray(scores, dtype=float),
        "rel": np.asarray(relevance, dtype=float),
        "qid": np.asarray(query_ids)
    })
    discounts = 1.0 / np.log2(np.arange(2, 7))

    def per_query(g):
        top = g.sort_values("score", ascending=False)["rel"].values[:5]
        ideal = np.sort(g["rel"].values)[::-1][:5]
        dcg = np.sum(top * discounts[:len(top)])
        idcg = np.sum(ideal * discounts[:len(ideal)])
        return dcg / idcg if idcg > 0 else 0.0

    return df_eval.groupby("qid", sort=False).apply(per_query).mean()

# -------------------------
# 2) Required notebook objects
# -------------------------
val = get_df("val", "dfval")
test = get_df("test", "dftest")

qid_col = get_col(val, "srch_id", "srchid")
prop_col = get_col(val, "prop_id", "propid")
brand_col = get_col(val, "prop_brand_bool", "propbrandbool")
click_col = get_col(val, "click_bool", "clickbool")
book_col = get_col(val, "booking_bool", "bookingbool")

test_qid_col = get_col(test, "srch_id", "srchid")
test_prop_col = get_col(test, "prop_id", "propid")
test_brand_col = get_col(test, "prop_brand_bool", "propbrandbool")

if "relevance" not in val.columns:
    val = val.copy()
    val["relevance"] = np.where(val[book_col] == 1, 5,
                         np.where(val[click_col] == 1, 1, 0))
rel_col = "relevance"

# -------------------------
# 3) Rebuild ensemble scores if needed
# -------------------------
def get_weights():
    g = globals()
    if "best" in g:
        try:
            vals = g["best"]
            return float(vals[0]), float(vals[1]), float(vals[2])
        except Exception:
            pass
    return 0.0, 0.8, 0.2

w_lr, w_lgbm, w_svd = get_weights()

ensemble_val, val_source = get_array("ensemble_val", "ensembleval", "blend_val", "blendval")
ensemble_test, test_source = get_array("ensemble_test", "ensembletest", "blend_test", "blendtest")

if ensemble_val is None:
    lr_val, _ = get_array("lrval")
    lgbm_val, _ = get_array("lgbmval")
    svd_val, _ = get_array("svdval")

    if lgbm_val is None or svd_val is None:
        raise ValueError("Task 5 needs either ensemble_val/ensembleval or at least lgbmval and svdval.")

    z_lgbm_val = per_query_zscore(lgbm_val, val[qid_col].values)
    z_svd_val = per_query_zscore(svd_val, val[qid_col].values)
    z_lr_val = per_query_zscore(lr_val, val[qid_col].values) if lr_val is not None else np.zeros(len(val), dtype=float)

    ensemble_val = w_lr * z_lr_val + w_lgbm * z_lgbm_val + w_svd * z_svd_val
    val_source = f"recomputed ensemble (LR={w_lr:.1f}, LGBM={w_lgbm:.1f}, SVD={w_svd:.1f})"

if ensemble_test is None:
    lr_test, _ = get_array("lrtest")
    lgbm_test, _ = get_array("lgbmtest")
    svd_test, _ = get_array("svdtest")

    if lgbm_test is None or svd_test is None:
        raise ValueError("Task 5 needs either ensemble_test/ensembletest or at least lgbmtest and svdtest.")

    z_lgbm_test = per_query_zscore(lgbm_test, test[test_qid_col].values)
    z_svd_test = per_query_zscore(svd_test, test[test_qid_col].values)
    z_lr_test = per_query_zscore(lr_test, test[test_qid_col].values) if lr_test is not None else np.zeros(len(test), dtype=float)

    ensemble_test = w_lr * z_lr_test + w_lgbm * z_lgbm_test + w_svd * z_svd_test
    test_source = f"recomputed ensemble (LR={w_lr:.1f}, LGBM={w_lgbm:.1f}, SVD={w_svd:.1f})"

print(f"Validation score source: {val_source}")
print(f"Test score source: {test_source}")

# -------------------------
# 4) Ranking / fairness helpers
# -------------------------
def build_rank_df(df, scores, score_name, qid_name, prop_name, brand_name, rel_name=None):
    cols = [qid_name, prop_name, brand_name]
    if rel_name is not None and rel_name in df.columns:
        cols.append(rel_name)

    out = df[cols].copy()
    out[score_name] = np.asarray(scores, dtype=float)
    out["is_independent"] = (out[brand_name] == 0).astype(np.int8)
    out["is_chain"] = (out[brand_name] == 1).astype(np.int8)
    out["rank"] = (
        out.groupby(qid_name)[score_name]
        .rank(method="first", ascending=False)
        .astype(float)
    )
    out["in_top5"] = (out["rank"] <= 5).astype(np.int8)
    return out

def discounted_exposure(rank_values, k=5):
    r = np.asarray(rank_values, dtype=float)
    return np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0)

def fairness_metrics(rank_df, score_col, qid_name, rel_name, k=5):
    df = rank_df.copy()
    df["disc_exposure"] = discounted_exposure(df["rank"].values, k=k)

    candidate_share_ind = df["is_independent"].mean()

    topk = df[df["in_top5"] == 1].copy()
    topk_share_ind = topk["is_independent"].mean()

    total_disc = topk["disc_exposure"].sum()
    disc_share_ind = (
        topk.loc[topk["is_independent"] == 1, "disc_exposure"].sum() / total_disc
        if total_disc > 0 else np.nan
    )

    avg_rank_ind = df.loc[df["is_independent"] == 1, "rank"].mean()
    avg_rank_chain = df.loc[df["is_chain"] == 1, "rank"].mean()

    rep_gap_ind = topk_share_ind - candidate_share_ind
    disc_gap_ind = disc_share_ind - candidate_share_ind if pd.notna(disc_share_ind) else np.nan

    ndcg5 = ndcg_at_5(df[score_col].values, df[rel_name].values, df[qid_name].values)

    return {
        "candidate_share_independent": float(candidate_share_ind),
        "top5_share_independent": float(topk_share_ind),
        "discounted_top5_share_independent": float(disc_share_ind),
        "representation_gap_independent": float(rep_gap_ind),
        "discounted_gap_independent": float(disc_gap_ind) if pd.notna(disc_gap_ind) else np.nan,
        "avg_rank_independent": float(avg_rank_ind),
        "avg_rank_chain": float(avg_rank_chain),
        "ndcg5": float(ndcg5),
    }

def query_underexposure(rank_df, qid_name, score_col="base_score", k=5):
    tmp = rank_df[[qid_name, "is_independent", score_col]].copy()
    tmp["rank_tmp"] = (
        tmp.groupby(qid_name)[score_col]
        .rank(method="first", ascending=False)
        .astype(float)
    )

    candidate_share = tmp.groupby(qid_name)["is_independent"].mean()
    topk_share = (
        tmp[tmp["rank_tmp"] <= k]
        .groupby(qid_name)["is_independent"]
        .mean()
    )

    topk_share = candidate_share.index.to_series().map(topk_share).fillna(0.0)
    gap = (candidate_share - topk_share).clip(lower=0.0)
    return gap

def apply_query_level_fair_rerank(rank_df, alpha, query_gap, qid_name, base_score_col="base_score", out_col="fair_score", k=5):
    out = rank_df.copy()
    out["query_underexposure"] = out[qid_name].map(query_gap).fillna(0.0).astype(float)

    # Query-aware bonus:
    # larger boost only in queries where independents are under-exposed
    out[out_col] = (
        out[base_score_col]
        + float(alpha) * out["query_underexposure"] * out["is_independent"]
    )

    out["rank"] = (
        out.groupby(qid_name)[out_col]
        .rank(method="first", ascending=False)
        .astype(float)
    )
    out["in_top5"] = (out["rank"] <= k).astype(np.int8)
    return out

# -------------------------
# 5) Validation fairness analysis
# -------------------------
val_rank = build_rank_df(val, ensemble_val, "base_score", qid_col, prop_col, brand_col, rel_col)
baseline = fairness_metrics(val_rank, "base_score", qid_col, rel_col, k=5)
baseline_ndcg = baseline["ndcg5"]

# Query-level under-exposure computed from the base ranking
val_query_gap = query_underexposure(val_rank, qid_col, score_col="base_score", k=5)

alpha_grid = np.round(np.arange(0.00, 1.01, 0.05), 2)
rows = []

for alpha in alpha_grid:
    reranked = apply_query_level_fair_rerank(
        val_rank, alpha, val_query_gap, qid_name=qid_col,
        base_score_col="base_score", out_col="fair_score", k=5
    )
    m = fairness_metrics(reranked, "fair_score", qid_col, rel_col, k=5)
    m["alpha"] = float(alpha)
    m["ndcg_drop"] = float(baseline_ndcg - m["ndcg5"])
    m["abs_representation_gap_independent"] = abs(m["representation_gap_independent"])
    m["mean_query_underexposure_before"] = float(val_query_gap.mean())
    rows.append(m)

grid = pd.DataFrame(rows)

# Pick the fairest alpha under a small NDCG loss tolerance
feasible = grid[grid["ndcg_drop"] <= 0.005].copy()

if len(feasible) > 0:
    best_row = feasible.sort_values(
        ["abs_representation_gap_independent", "ndcg_drop", "alpha"],
        ascending=[True, True, True]
    ).iloc[0]
else:
    best_row = grid.sort_values(
        ["abs_representation_gap_independent", "ndcg_drop", "alpha"],
        ascending=[True, True, True]
    ).iloc[0]

best_alpha = float(best_row["alpha"])

val_rank_fair = apply_query_level_fair_rerank(
    val_rank, best_alpha, val_query_gap, qid_name=qid_col,
    base_score_col="base_score", out_col="fair_score", k=5
)
after = fairness_metrics(val_rank_fair, "fair_score", qid_col, rel_col, k=5)

# -------------------------
# 6) Save report outputs
# -------------------------
summary = pd.DataFrame([
    {
        "stage": "before_mitigation",
        "method": "base_ensemble",
        "alpha": 0.0,
        **baseline,
        "ndcg_drop": 0.0,
        "mean_query_underexposure_before": float(val_query_gap.mean())
    },
    {
        "stage": "after_mitigation",
        "method": "query_level_adaptive_rerank",
        "alpha": best_alpha,
        **after,
        "ndcg_drop": baseline["ndcg5"] - after["ndcg5"],
        "mean_query_underexposure_before": float(val_query_gap.mean())
    },
])

report_values = pd.DataFrame([{
    "chosen_alpha": best_alpha,
    "baseline_ndcg5": baseline["ndcg5"],
    "mitigated_ndcg5": after["ndcg5"],
    "independent_candidate_share": baseline["candidate_share_independent"],
    "independent_top5_share_before": baseline["top5_share_independent"],
    "independent_top5_share_after": after["top5_share_independent"],
    "independent_discounted_top5_share_before": baseline["discounted_top5_share_independent"],
    "independent_discounted_top5_share_after": after["discounted_top5_share_independent"],
    "representation_gap_before": baseline["representation_gap_independent"],
    "representation_gap_after": after["representation_gap_independent"],
    "discounted_gap_before": baseline["discounted_gap_independent"],
    "discounted_gap_after": after["discounted_gap_independent"],
    "avg_rank_independent_before": baseline["avg_rank_independent"],
    "avg_rank_independent_after": after["avg_rank_independent"],
    "avg_rank_chain_before": baseline["avg_rank_chain"],
    "avg_rank_chain_after": after["avg_rank_chain"],
    "mean_query_underexposure_before": float(val_query_gap.mean()),
}])

os.makedirs("outputs", exist_ok=True)
summary.to_csv("outputs/task5_query_fairness_summary.csv", index=False)
grid.to_csv("outputs/task5_query_fairness_grid.csv", index=False)
report_values.to_csv("outputs/task5_query_report_values.csv", index=False)

print("\n=== TASK 5 SUMMARY ===")
print(summary.round(4).to_string(index=False))

print("\n=== BEST ALPHA ===")
print(best_row.round(4).to_string())

print("\n=== REPORT VALUES ===")
print(report_values.round(4).to_string(index=False))

print("\n=== LATEX REPLACEMENTS ===")
vals = report_values.iloc[0].to_dict()
for k, v in vals.items():
    print(f"{k} = {v:.4f}")

# -------------------------
# 7) Apply the same reranking to the test set
# -------------------------
test_rank = build_rank_df(test, ensemble_test, "base_score", test_qid_col, test_prop_col, test_brand_col, rel_name=None)
test_query_gap = query_underexposure(test_rank, test_qid_col, score_col="base_score", k=5)

test_rank_fair = apply_query_level_fair_rerank(
    test_rank, best_alpha, test_query_gap, qid_name=test_qid_col,
    base_score_col="base_score", out_col="fair_score", k=5
)

submission_task5 = (
    test_rank_fair[[test_qid_col, test_prop_col, "fair_score"]]
    .sort_values([test_qid_col, "fair_score"], ascending=[True, False], kind="mergesort")
    [[test_qid_col, test_prop_col]]
)

submission_task5.to_csv("outputs/submission_task5_query_fair.csv", index=False)

print(f"\nsubmission_task5_query_fair.csv written: {len(submission_task5):,} rows")
print(f"Unique searches: {submission_task5[test_qid_col].nunique():,}")