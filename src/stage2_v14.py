"""Stage 2 v14: the v10 decoy-aware stage-2 matcher + extra features (extra_feats.py) + more training data (T2).

  train   : training table = V + T (s2v2_train.parquet) + T2 (s2v2_train_T2.parquet, prep_T2.py).
            Decoy-word scores are re-learned on the combined table, extra features are added.
            Local check (US+India, local_eval.py): cross-fit T->V / V->T with and without T2, then the final model
            on V+T+T2 (negatives x2.6, the test decoy density).
  predict : like stage2_v2.predict (cluster context, decoy features, decoy-word scores, prior adjustment per country,
            expected-F0.5 selection) + the extra features. Writes <work>/test_best_s2v2_cf_final.parquet (the file the
            later steps read) and <out>/matching_results.tsv, <out>/candidate_pairs.tsv.

usage: python stage2_v14.py <work_dir> <models_dir> {train|predict} [out_dir]
"""
import gc
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import stage2
import decoy_features as D
import extra_feats as E
from train import read_part

W, MD, STEP = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 and STEP != "predict_country" else f"{W}/out_v14"
t0 = time.time()
log = lambda *a: print(*a, f"{time.time()-t0:.0f}s", flush=True)
P1_MIN = 0.02
RATIO = 2.6
DW_COLS = ["dw_max", "dw_sum", "dw_n_hi"]
QCOLS = ["rid", "entity_id", "country", "ncore", "nfull", "atoks", "anums", "acity"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=17)


def add_dw_features(best, q, s1):
    ops = D.op_tokens(best.select("rid", "s1"), q, s1)
    weak = best.select("rid", "s1", "country",
                       ((pl.col("d1_strong") == 1) | ((pl.col("one_repl") == 1) & (pl.col("dd_strong") == 1))).fill_null(False).alias("weak_decoy"),
                       (pl.col("d1") == 0).fill_null(False).alias("weak_true"))
    sc = D.decoy_word_scores(ops, weak)
    f = D.decoy_word_features(ops, sc, weak.select("rid", "s1", "country"))
    return best.drop(DW_COLS, strict=False).join(f, on=["rid", "s1"], how="left")


def ev_select(best, col="p2", max_n=12, fallback_thr=0.8):
    from local_eval import ev_select as _ev
    return _ev(best, col, max_n, fallback_thr)


def training_table():
    path = f"{W}/s2v2_train_v14.parquet"
    if os.path.exists(path):
        return pl.read_parquet(path)
    a = pl.read_parquet(f"{W}/s2v2_train.parquet").with_columns(pl.lit(False).alias("isT2")).drop(DW_COLS, strict=False)
    qc = pl.read_parquet(f"{W}/train_q.parquet", columns=["rid", "country"])
    extra = []
    for pth in sorted(glob.glob(f"{W}/s2v2_train_T[0-9].parquet")):   # T2, T3 ... (prep_T2.py): training rows only
        b = pl.read_parquet(pth).drop("country", strict=False).join(qc, on="rid", how="left").filter(pl.col("p1") >= P1_MIN)
        extra.append(b.select([c for c in a.columns if c in b.columns and c != "isT2"] + ["isT2"]))
        log("extra training rows", pth, b.height)
    best = pl.concat([a.filter(pl.col("p1") >= P1_MIN)] + extra, how="diagonal_relaxed")
    del a, extra, b
    best = best.with_columns([pl.col(c_).cast(pl.Float32) for c_, dt in best.schema.items() if dt == pl.Float64])
    gc.collect()
    q = pl.read_parquet(f"{W}/train_q.parquet", columns=["rid", "entity_id", "ncore", "anums", "atoks", "acity", "country"]).join(best.select("rid").unique(), on="rid")
    s1 = pl.read_parquet(f"{W}/train_s1.parquet", columns=["rid", "ncore", "anums", "atoks", "acity"]).join(best.select(pl.col("s1").alias("rid")).unique(), on="rid")
    best = add_dw_features(best, q.select("rid", "ncore", "country"), s1.select("rid", "ncore"))
    ex = E.compute(best.select("rid", "s1", "p1"), q, s1)
    best = best.join(ex, on=["rid", "s1"], how="left")
    best.write_parquet(path)
    log("training table", best.height, "V", int(best["isV"].sum()), "T2", int(best["isT2"].sum()))
    return best


def fit(path, cols, row_filter, rounds=None, seed=17):
    """LightGBM on the rows of the cached training table that pass row_filter. Memory-light: the feature matrix is
    filled column group by column group straight from parquet (no DataFrame + matrix copies), with the
    early-stopping rows placed at the end so both parts are views."""
    params = dict(PARAMS, seed=seed)
    base = pl.scan_parquet(path).filter(row_filter)
    key = base.select("label", "rid").collect()
    y = key["label"].to_numpy()
    es = (key["rid"].hash(seed=3) % 10 == 0).to_numpy() if rounds is None else np.zeros(key.height, dtype=bool)
    del key
    order = np.argsort(es, kind="stable")          # early-stopping rows last
    n_tr = int((~es).sum())
    y, es_sorted = y[order], es[order]
    wt = np.where(y == 1, 1.0, RATIO).astype(np.float32)
    X = np.empty((len(order), len(cols)), dtype=np.float32)
    G = 16
    for g in range(0, len(cols), G):
        part = base.select(cols[g:g + G]).collect().to_numpy().astype(np.float32, copy=False)
        X[:, g:g + G] = part[order]
        del part
    gc.collect()
    if rounds is None:
        dtr = lgb.Dataset(X[:n_tr], label=y[:n_tr], weight=wt[:n_tr], feature_name=cols, free_raw_data=True)
        dva = lgb.Dataset(X[n_tr:], label=y[n_tr:], weight=wt[n_tr:], reference=dtr)
        dtr.construct(); dva.construct()
        del X
        gc.collect()
        return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    d = lgb.Dataset(X, label=y, weight=wt, feature_name=cols, free_raw_data=True)
    d.construct()
    del X
    gc.collect()
    return lgb.train(params, d, num_boost_round=rounds)


def train():
    from local_eval import LocalEval
    path = f"{W}/s2v2_train_v14.parquet"
    if not os.path.exists(path):
        training_table()
        gc.collect()
    meta10 = json.load(open(f"{W}/s2v2_results.json"))
    cols = meta10["cf_cols"] + E.EXTRA_FEATS
    le = LocalEval(W)
    preds, its = [], []
    for fold, tr_mask in (("T->V", ~pl.col("isV") & ~pl.col("isT2")), ("V->T", pl.col("isV"))):
        m = fit(path, cols, tr_mask | pl.col("isT2"))
        its.append(m.best_iteration)
        te = pl.scan_parquet(path).filter(~pl.col("isT2") & ~tr_mask).select(list(dict.fromkeys(cols + ["rid", "s1", "p1", "label"]))).collect()
        preds.append(te.select("rid", "s1", "p1", "label").with_columns(pl.Series("p2", m.predict(te.select(cols).to_numpy().astype(np.float32), num_threads=2))))
        log("extra-subsets", fold, "iters", m.best_iteration)
        del m, te
        gc.collect()
    pv = pl.concat(preds)
    pv.write_parquet(f"{W}/cf_preds_v14_T2.parquet")
    log("local (cross-fit, all extra training subsets):")
    le.score(pv.select("rid", "s1", "p2"))
    rounds = int(np.mean(its) * 1.1 * 1.15)   # the final model sees ~1.3x the rows of a cross-fit fold
    m = fit(path, cols, pl.lit(True), rounds)
    m.save_model(f"{W}/s2v2_cf_v14.txt")
    best = pl.read_parquet(path, columns=["label"])
    nS1 = pl.concat([pl.read_parquet(f"{W}/val_s1.parquet"), pl.read_parquet(f"{W}/T_s1.parquet")]
                    + [pl.read_parquet(p_) for p_ in sorted(glob.glob(f"{W}/T[0-9]_s1.parquet"))]).unique().height
    meta = {"cols": cols, "ratio": RATIO, "rounds": rounds, "iters": its,
            "train_pos_per_s1": float(best["label"].sum() / nS1), "train_neg_per_s1": float((best.height - best["label"].sum()) / nS1)}
    json.dump(meta, open(f"{W}/s2v2_v14.json", "w"), indent=1)
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda t: -t[1])[:25]
    log("final model rounds", rounds, "top features", [(c, int(g)) for c, g in imp])


def predict_country(c):
    """stage-2 scores for the test best pairs of one country -> <work>/test_best_v14_<c>.parquet (own process: memory)."""
    meta = json.load(open(f"{W}/s2v2_v14.json"))
    cols = meta["cols"]
    m2 = lgb.Booster(model_file=f"{W}/s2v2_cf_v14.txt")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet")
    s1dups = stage2.s1_duplicates(s1.select("rid", "country", "ncore", "atoks"))
    vocab = D.s1_vocab(s1)
    pred_path = f"{W}/pred_test_cf.parquet"   # stage-1 test scores (stage2_v2.predict writes it)
    q = pl.scan_parquet(f"{W}/test_q.parquet").filter(pl.col("country") == c).select(QCOLS).collect()
    s1 = s1.join(pl.scan_parquet(pred_path).join(q.select("rid").lazy(), on="rid").select(pl.col("s1").alias("rid")).unique().collect(), on="rid")
    gc.collect()
    pr = pl.scan_parquet(pred_path).join(q.select("rid").lazy(), on="rid").collect()
    cf = pl.concat([read_part(p).select("rid", "s1", "a_tset", "nf_tset") for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet"))])
    b = stage2.build(pr, q.select("rid", "entity_id", "ncore", "atoks", "anums"), cf, s1dups)
    del cf, pr, s1dups
    gc.collect()
    b = b.filter(pl.col("p1") >= P1_MIN)
    f = D.compute(b.select("rid", "s1"), q, s1, vocab)
    cl = D.cluster_features(b.select("rid", "s1", "p1"), f.select("rid", "s1", "d1"), q)
    b = b.join(f, on=["rid", "s1"], how="left").join(cl, on=["rid", "s1"], how="left")
    del f, cl
    gc.collect()
    b = add_dw_features(b.with_columns(pl.lit(c).alias("country")), q, s1)
    exp = f"{W}/test_extra_{c}.parquet"   # extra_test.py (same best pairs, computed in its own process)
    if os.path.exists(exp):
        ex = pl.read_parquet(exp)
    else:
        ex = E.compute(b.select("rid", "s1", "p1"), q.select("rid", "entity_id", "anums", "atoks", "acity", "ncore"),
                       s1.select("rid", "anums", "atoks", "acity", "ncore"))
    del q, s1
    gc.collect()
    b = b.join(ex, on=["rid", "s1"], how="left")
    del ex
    b = b.with_columns([pl.col(c_).cast(pl.Float32) for c_, dt in b.schema.items() if dt == pl.Float64])
    gc.collect()
    outs = []
    for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet")):
        f = b.join(read_part(p).drop("p1", strict=False), on=["rid", "s1"], how="inner")
        if f.height:
            outs.append(f.select("rid", "s1", "p1").with_columns(
                pl.Series("p2", m2.predict(f.select(cols).to_numpy().astype(np.float32), num_threads=2)).cast(pl.Float32)))
        del f
    pl.concat(outs).write_parquet(f"{W}/test_best_v14_{c}.parquet")
    log("predicted", c)


def predict():
    """runs predict_country for every country in a separate process, then the prior adjustment and the selection."""
    import shutil
    import subprocess
    meta = json.load(open(f"{W}/s2v2_v14.json"))
    v10_path = f"{W}/test_best_s2v2_cf_final_v10.parquet"
    if not os.path.exists(v10_path) and os.path.exists(f"{W}/test_best_s2v2_cf_final.parquet"):
        shutil.copyfile(f"{W}/test_best_s2v2_cf_final.parquet", v10_path)   # the v10 scores (stage2_v2.py predict)
    os.makedirs(OUT, exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "country"])
    for c in s1["country"].unique().sort().to_list():
        if not os.path.exists(f"{W}/test_best_v14_{c}.parquet"):
            subprocess.run([sys.executable, os.path.abspath(__file__), W, MD, "predict_country", c], check=True)
    best = pl.concat([pl.read_parquet(f"{W}/test_best_v14_{c}.parquet") for c in s1["country"].unique().to_list()])
    # per-country label-shift correction (as stage2_v2.predict with PRIOR_ADJ=1)
    cc = s1.select(pl.col("rid").alias("s1"), "country")
    nS1 = cc.group_by("country").len().rename({"len": "n"})
    cnt = best.join(cc, on="s1").group_by("country").len().rename({"len": "cand"}).join(nS1, on="country")
    pos, neg, r0 = meta["train_pos_per_s1"], meta["train_neg_per_s1"], float(meta["ratio"])
    cnt = cnt.with_columns(((pl.col("cand") / pl.col("n") - pos) / neg).alias("ratio_c")).with_columns((r0 / pl.col("ratio_c")).alias("k"))
    ks = float(os.environ.get("KSCALE", "1"))   # optional extra odds factor for US and India (local harness optimum 1.5)
    if ks != 1.0:
        cnt = cnt.with_columns(pl.when(pl.col("country").is_in(["US", "India"])).then(pl.col("k") * ks).otherwise(pl.col("k")).alias("k"))
    log("prior adjustment per country", cnt.select("country", "ratio_c", "k").to_dicts())
    best = (best.join(cc, on="s1").join(cnt.select("country", "k"), on="country")
                .with_columns(pl.col("p2").alias("p2_raw"),
                              (pl.col("p2") * pl.col("k") / (pl.col("p2") * pl.col("k") + 1 - pl.col("p2"))).alias("p2"))
                .drop("country", "k"))
    # Decoy groups: in the test, decoy entities often have 2+ records at the same shifted house number (strong-offset
    # pairs sharing their number with another record of the entity: US 19% vs 1.2% for train decoys). A model trained on
    # train data reads "shared deviation" as a true-record signal, so for pairs whose number is shifted UP by a decoy
    # offset the v14 score may not exceed the v10 score (symmetry check on the test: +k accepted pairs vs -k).
    if os.path.exists(v10_path):
        d1 = pl.concat([pl.read_parquet(p, columns=["rid", "s1", "d1"]) for p in sorted(glob.glob(f"{W}/test_best_feats_*.parquet"))])
        v10 = pl.read_parquet(v10_path).select("rid", "s1", pl.col("p2").alias("p2_v10"))
        best = (best.join(d1, on=["rid", "s1"], how="left").join(v10, on=["rid", "s1"], how="left")
                    .with_columns(pl.when(pl.col("d1").is_in([3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 21.0]) & pl.col("p2_v10").is_not_null())
                                    .then(pl.min_horizontal("p2", "p2_v10")).otherwise(pl.col("p2")).alias("p2"))
                    .drop("d1", "p2_v10"))
        log("capped strong positive offsets at the v10 score")
    best.write_parquet(f"{W}/test_best_s2v2_cf_final.parquet")
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
    acc = ev_select(best, "p2").join(qid, on="rid")
    match = (s1id.join(acc.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
                 .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    cand = best.select("rid", "s1").join(qid, on="rid")
    cp = (s1id.join(cand.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("candidate_entity_ids")), on="s1", how="left")
              .select("source1_entity_id", pl.col("candidate_entity_ids").fill_null("")))
    cp.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    log("written", OUT, "matched pairs", acc.height)


if __name__ == "__main__":
    if STEP == "predict_country":
        predict_country(sys.argv[4])
    else:
        {"train": train, "predict": predict}[STEP]()
