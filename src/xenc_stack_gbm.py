"""v20 stacker: LightGBM on [logit p2, both cross-encoder logits, logit p1, country, pair decoy/noise features] for the
uncertain band (0.002 < p2 < 0.998), trained on the held-out V+T band pairs (negatives x2.6, same scale as p2).
Local US+India (2-fold by Source-1): logistic stacker 0.98969 (v19, K x1.5) -> LightGBM 0.98978 (K x1.5) / 0.98981 (K x1.75).
The pair features are the stage-2 decoy/noise features of the best pair (no cluster-context features: test decoy groups).

usage:
  python xenc_stack_gbm.py <work> eval  <xenc1.csv,xenc2.csv>             # local check
  python xenc_stack_gbm.py <work> apply <xenc1.csv,xenc2.csv> <out_dir>   # test (env KSCALE, default 1.75; V15_DIR = v16 folder)
"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_eval import LocalEval, ev_select, K_V10  # noqa: E402

W, STEP, XS = sys.argv[1], sys.argv[2], sys.argv[3]
XFILES = [f for f in XS.split(",") if f]
XC = [f"xenc{i}" for i in range(len(XFILES))]
V15 = os.environ.get("V15_DIR", f"{W}/v16")
KS = float(os.environ.get("KSCALE", "1.75"))
FE = ["d1", "d1_strong", "d1_possmall_other", "name_exact_full", "name_exact_core", "legal_swapped", "legal_added", "legal_removed",
      "n_add", "n_del", "n_typo", "street_eq", "street_jac", "city_overlap", "q_all_novel", "initials_match", "max_typo_dist",
      "first_tok_typo", "n_add_pseudo"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=1)
ROUNDS = 500


def lg(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def scores(split):
    out = None
    for i, f in enumerate(XFILES):
        d = (pl.read_csv(f, schema_overrides={"rid": pl.UInt32, "s1": pl.UInt32, "score": pl.Float64})
               .filter(pl.col("split") == split).select("rid", "s1", pl.col("score").alias(f"xenc{i}")).unique(["rid", "s1"]))
        out = d if out is None else out.join(d, on=["rid", "s1"])
    return out


def mat(d, p2col):
    cols = [lg(d[p2col]), *[d[c].to_numpy() for c in XC], lg(d["p1"]), (d["country"] == "India").cast(pl.Float64).to_numpy()]
    cols += [d[f].cast(pl.Float64).fill_null(np.nan).to_numpy() for f in FE]
    return np.column_stack(cols)


def train_band():
    le = LocalEval(W)
    cf = pl.read_parquet(f"{V15}/cf_preds_v14_T2.parquet")
    band = cf.join(scores("vt"), on=["rid", "s1"]).join(le.s1.select("s1", "country"), on="s1")
    ft = pl.scan_parquet(f"{W}/s2v2_train_v14.parquet").select(["rid", "s1"] + FE).join(band.select("rid", "s1").lazy(), on=["rid", "s1"]).collect()
    return le, cf, band.join(ft, on=["rid", "s1"], how="left")


def eval_():
    le, cf, band = train_band()
    X, y = mat(band, "p2"), band["label"].to_numpy()
    w = np.where(y == 1, 1.0, 2.6)
    fold = (band["s1"].hash(seed=5) % 2).to_numpy()
    out = np.empty(len(y))
    for k in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(X[fold != k], y[fold != k], weight=w[fold != k]), num_boost_round=ROUNDS)
        out[fold == k] = m.predict(X[fold == k])
    pv = cf.join(band.select("rid", "s1").with_columns(pl.Series("p2x", out)), on=["rid", "s1"], how="left").with_columns(pl.col("p2x").fill_null(pl.col("p2")))
    for ks in (1.5, 1.75, 2.0):
        kk = {c: K_V10[c] * ks for c in K_V10}
        r0 = le.score(pv, "p2", k=kk, verbose=False)["US+India"]
        r1 = le.score(pv, "p2x", k=kk, verbose=False)["US+India"]
        print(f"K x{ks}: without cross-encoders {r0:.5f}  with LightGBM stacker {r1:.5f}  gain {r1 - r0:+.5f}")


def apply_(out):
    le, cf, band = train_band()
    X, y = mat(band, "p2"), band["label"].to_numpy()
    m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=np.where(y == 1, 1.0, 2.6)), num_boost_round=ROUNDS)
    os.makedirs(out, exist_ok=True)
    m.save_model(f"{out}/xenc_stack_gbm.txt")
    best = pl.read_parquet(f"{V15}/test_best_s2v2_cf_final.parquet")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "country"])
    cc = s1.select(pl.col("rid").alias("s1"), "country")
    meta = json.load(open(f"{V15}/s2v2_v14.json"))
    nS1 = cc.group_by("country").len().rename({"len": "n"})
    cnt = best.join(cc, on="s1").group_by("country").len().rename({"len": "cand"}).join(nS1, on="country")
    pos, neg, r0 = meta["train_pos_per_s1"], meta["train_neg_per_s1"], float(meta["ratio"])
    cnt = cnt.with_columns(((pl.col("cand") / pl.col("n") - pos) / neg).alias("ratio_c")).with_columns((r0 / pl.col("ratio_c")).alias("k"))
    cnt = cnt.with_columns(pl.when(pl.col("country").is_in(["US", "India"])).then(pl.col("k") * KS).otherwise(pl.col("k")).alias("k"))
    print(cnt.select("country", "ratio_c", "k").to_dicts())
    feats = pl.concat([pl.read_parquet(p, columns=["rid", "s1"] + FE) for p in sorted(glob.glob(f"{W}/test_best_feats_*.parquet"))])
    b = (best.join(scores("test"), on=["rid", "s1"], how="left").join(cc, on="s1").join(cnt.select("country", "k"), on="country")
             .join(feats, on=["rid", "s1"], how="left"))
    use = pl.Series([True] * b.height)
    for c in XC:
        use = use & b[c].is_not_null()
    use = (use & b["country"].is_in(["US", "India"])).to_numpy()
    print("test pairs", b.height, "re-scored", int(use.sum()))
    newraw = b["p2_raw"].to_numpy().astype(np.float64)
    bu = b.filter(pl.Series(use))
    newraw[use] = m.predict(mat(bu, "p2_raw"))
    kk = b["k"].to_numpy()
    adj = newraw * kk / (newraw * kk + 1 - newraw)
    v10 = pl.read_parquet(f"{W}/test_best_s2v2_cf_final_v10.parquet").select("rid", "s1", pl.col("p2").alias("p2_v10"))
    b = b.join(v10, on=["rid", "s1"], how="left")
    cap_c = [c.strip() for c in os.environ.get("CAP_COUNTRIES", "US,India,France").split(",")]
    cap = (b["d1"].is_in([3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 21.0]) & b["p2_v10"].is_not_null() & b["country"].is_in(cap_c)).fill_null(False).to_numpy()
    new_p2 = np.where(cap, np.minimum(adj, b["p2_v10"].fill_null(1.0).to_numpy()), adj)
    print("capped pairs", int(cap.sum()))
    b = b.select("rid", "s1", "p1", "p2_raw", *XC, pl.col("p2").alias("p2_base"), pl.Series("p2", new_p2), pl.Series("p2_nocap", adj))
    b.write_parquet(f"{out}/test_best_xenc.parquet")
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
    acc = ev_select(b, "p2").join(qid, on="rid")
    print("accepted pairs", acc.height)
    match = (s1id.join(acc.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
                 .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    match.write_csv(f"{out}/matching_results.tsv", separator="\t", quote_style="never")
    cand = b.select("rid", "s1").join(qid, on="rid")
    cp = (s1id.join(cand.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("candidate_entity_ids")), on="s1", how="left")
              .select("source1_entity_id", pl.col("candidate_entity_ids").fill_null("")))
    cp.write_csv(f"{out}/candidate_pairs.tsv", separator="\t", quote_style="never")
    print("written", out)


if __name__ == "__main__":
    {"eval": lambda: eval_(), "apply": lambda: apply_(sys.argv[4])}[STEP]()
