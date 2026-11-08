"""Stage 2 v2: decoy-aware acceptance model, trained and tuned at TEST-LIKE decoy density.

Why: the test set has about twice as many decoy records per Source-1 entity as train, and the
decoys follow a fixed recipe (legal suffix swapped/added, a real word added or substituted, one
address number shifted up by +1..+21). The original stage 2 sees neither the recipe nor the
density, so it accepts too many decoys and, being conservative, rejects messy true matches.

What this script does
  prep_train : out-of-fold stage-1 scores for the V (3%) and T (4%) Source-1 subsets, using the
               cross-fitted stage-1 fold models (model k scores the fold it did not train on),
               best pair per query, the original stage-2 context features, plus the new
               decoy/noise features (decoy_features.py).
  train      : LightGBM on T best pairs with negatives up-weighted by the test/train decoy ratio;
               evaluated on V with negatives duplicated by the same ratio (test-like), against the
               same model without the new features. Picks the threshold on the test-like V score.
  predict    : scores the test best pairs and writes matching_results.tsv / candidate_pairs.tsv.

usage: python stage2_v2.py <work_dir> <models_dir> {prep_train|train|predict} [out_dir]
"""
import glob
import json
import os
import sys
import time
import gc

import numpy as np
import polars as pl
import lightgbm as lgb

import stage2
import decoy_features as D
from train import read_part
from metrics import macro_f05

W, MD, STEP = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else f"{W}/out_v2"
t0 = time.time()
log = lambda *a: print(*a, f"{time.time()-t0:.0f}s", flush=True)
P1_MIN = 0.02
NEW_CTX = ["ctx_s1num_confirmed", "ctx_same_offset", "ctx_same_name", "ctx_n"]
QCOLS = ["rid", "entity_id", "country", "ncore", "nfull", "atoks", "anums", "acity"]


DW_COLS = ["dw_max", "dw_sum", "dw_n_hi"]


def add_dw_features(best: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Self-supervised decoy-word scores learned on this table itself (no labels): tokens added to the
    record name are scored by how often they occur in pairs with the shifted-number fingerprint vs pairs
    with the same house number, per country. best must carry rid, s1, country, d1, d1_strong, one_repl, dd_strong."""
    ops = D.op_tokens(best.select("rid", "s1"), q, s1)
    weak = best.select("rid", "s1", "country",
                       ((pl.col("d1_strong") == 1) | ((pl.col("one_repl") == 1) & (pl.col("dd_strong") == 1))).fill_null(False).alias("weak_decoy"),
                       (pl.col("d1") == 0).fill_null(False).alias("weak_true"))
    sc = D.decoy_word_scores(ops, weak)
    f = D.decoy_word_features(ops, sc, weak.select("rid", "s1", "country"))
    return best.drop(DW_COLS, strict=False).join(f, on=["rid", "s1"], how="left")


def stage1_models():
    meta = json.load(open(f"{MD}/model_stage1_cf.json"))
    return meta["cols"], [lgb.Booster(model_file=f"{MD}/model_stage1_f{k}.txt") for k in (0, 1)]


def add_new_features(best: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, vocab: dict) -> pl.DataFrame:
    f = D.compute(best.select("rid", "s1"), q, s1, vocab)
    c = D.cluster_features(best.select("rid", "s1", "p1"), f.select("rid", "s1", "d1"), q)
    return best.join(f, on=["rid", "s1"], how="left").join(c, on=["rid", "s1"], how="left")


def prep_train():
    cols, models = stage1_models()
    parts = sorted(glob.glob(f"{W}/feat/train_val_part*.parquet")) + sorted(glob.glob(f"{W}/feat/train_T_part*.parquet"))
    os.makedirs(f"{W}/s2v2_oof", exist_ok=True)
    seen = set()   # queries already scored (a query can appear in both V and T parts)
    for i, p in enumerate(parts):
        outp = f"{W}/s2v2_oof/part{i:04d}.parquet"
        if os.path.exists(outp):
            seen.update(pl.read_parquet(outp, columns=["rid"])["rid"].unique().to_list())
            continue
        d = read_part(p)
        if seen:
            d = d.filter(~pl.col("rid").is_in(pl.Series(list(seen), dtype=d.schema["rid"]).implode()))
        fold = (d["rid"].hash(seed=11) % 2).to_numpy()
        p1 = np.empty(d.height, dtype=np.float32)
        for k in (0, 1):
            m = fold == (1 - k)  # scored by the model that did NOT train on this fold
            if m.any():
                p1[m] = models[k].predict(d.filter(pl.Series(m)).select(cols).to_numpy().astype(np.float32), num_threads=2)
        d.select("rid", "s1", "label", "a_tset", "nf_tset").with_columns(pl.Series("p1", p1)).write_parquet(outp)
        seen.update(d["rid"].unique().to_list())
        del d; gc.collect()
        if i % 10 == 0:
            log("oof part", i, "of", len(parts))
    oof = pl.read_parquet(f"{W}/s2v2_oof/part*.parquet")
    log("oof", oof.height)
    s1 = pl.read_parquet(f"{W}/train_s1.parquet")
    s1dups = stage2.s1_duplicates(s1.select("rid", "country", "ncore", "atoks"))
    q = pl.read_parquet(f"{W}/train_q.parquet", columns=QCOLS).join(oof.select("rid").unique(), on="rid")
    vocab = D.s1_vocab(s1)
    s1V = pl.read_parquet(f"{W}/val_s1.parquet")
    s1T = pl.read_parquet(f"{W}/T_s1.parquet")
    outs = []
    for c in q["country"].unique().to_list():
        qc = q.filter(pl.col("country") == c)
        o = oof.join(qc.select("rid"), on="rid")
        b = stage2.build(o.select("rid", "s1", "p1"), qc.select("rid", "entity_id", "ncore", "atoks", "anums"),
                         o.select("rid", "s1", "a_tset", "nf_tset"), s1dups)
        # only best pairs whose S1 is in V or T have complete cluster context
        b = b.join(pl.concat([s1V, s1T]).unique(), on="s1")
        b = add_new_features(b, qc, s1, vocab)
        outs.append(b)
        log("built", c, b.height)
    best = pl.concat(outs, how="diagonal_relaxed")
    # stage-1 feature rows of the best pairs only (second pass over the parts)
    keys = best.select("rid", "s1")
    fr = []
    for p in parts:
        fr.append(read_part(p).join(keys, on=["rid", "s1"]).drop("label", "true_s1", strict=False))
    F = pl.concat(fr, how="diagonal_relaxed").unique(["rid", "s1"])
    del fr; gc.collect()
    best = best.join(F, on=["rid", "s1"], how="left")
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet")  # rid, true_s1
    best = best.join(gt, on="rid", how="left").with_columns((pl.col("true_s1") == pl.col("s1")).fill_null(False).cast(pl.Int8).alias("label"))
    best = best.with_columns(pl.col("s1").is_in(s1V["s1"].implode()).alias("isV"))
    best.write_parquet(f"{W}/s2v2_train.parquet")
    log("saved", best.height, "V rows", best["isV"].sum())


_FTAB = {}


def _ftable(n: int):
    """all 2^n outcomes (true/false per candidate, candidates sorted by p desc) and F0.5 of predicting the top-k."""
    if n in _FTAB:
        return _FTAB[n]
    z = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)   # (2^n, n)
    T = z.sum(1)
    F = np.zeros((2 ** n, n + 1))
    F[:, 0] = (T == 0).astype(float)
    for k in range(1, n + 1):
        tp = z[:, :k].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = tp / k
            R = np.where(T > 0, tp / T, 0.0)
            f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
        F[:, k] = f
    _FTAB[n] = (z, F)
    return z, F


def ev_select(best: pl.DataFrame, col: str = "p2", max_n: int = 12, fallback_thr: float = 0.8) -> pl.DataFrame:
    """Per Source-1 entity, accept the top-k candidates (by probability) that maximise the EXPECTED
    per-entity F0.5, treating candidate labels as independent Bernoulli(p). Returns accepted (rid, s1)."""
    b = best.select("rid", "s1", col).sort(["s1", col], descending=[False, True])
    g = b.group_by("s1", maintain_order=True).agg(pl.col("rid"), pl.col(col).alias("p"), pl.len().alias("n"))
    acc = []
    for n in sorted(g["n"].unique().to_list()):
        gg = g.filter(pl.col("n") == n)
        if n > max_n:
            x = gg.explode(["rid", "p"]).filter(pl.col("p") >= fallback_thr)
            acc.append(x.select("rid", "s1"))
            continue
        P = np.clip(np.array(gg["p"].to_list(), dtype=np.float64), 1e-6, 1 - 1e-6)   # (G, n)
        z, F = _ftable(n)
        logp = np.log(P) @ z.T + np.log(1 - P) @ (1 - z).T                           # (G, 2^n)
        prob = np.exp(logp)
        ef = prob @ F                                                                 # (G, n+1)
        k = ef.argmax(1)
        rids = gg["rid"].to_list(); s1s = gg["s1"].to_list()
        for i in np.nonzero(k)[0]:
            for r in rids[i][: k[i]]:
                acc.append((r, s1s[i]))
    tuples = [a for a in acc if isinstance(a, tuple)]
    frames = [a for a in acc if not isinstance(a, tuple)]
    out = pl.DataFrame(tuples, schema={"rid": best.schema["rid"], "s1": best.schema["s1"]}, orient="row") if tuples else None
    return pl.concat(([out] if out is not None else []) + frames)


def testlike_f05_pred(acc: pl.DataFrame, va: pl.DataFrame, gt_pairs: pl.DataFrame, s1V: pl.DataFrame, dup: int) -> float:
    a = acc.join(va.select("rid", "s1", "label"), on=["rid", "s1"])
    extra = [a.filter(pl.col("label") == 0).with_columns((pl.col("rid").cast(pl.Int64) + (4_000_000_000 - 20_000_000 * i)).cast(pl.UInt32).alias("rid"))
             for i in range(1, dup)]
    return macro_f05(pl.concat([a] + extra).select("s1", "rid"), gt_pairs, s1V)


def testlike_f05(va: pl.DataFrame, score: str, thr: float, gt_pairs: pl.DataFrame, s1V: pl.DataFrame, dup: int) -> float:
    """Macro F0.5 on V where every accepted NEGATIVE counts `dup` times (test has ~dup x decoys)."""
    acc = va.filter((pl.col(score) >= thr) & (pl.col("p1") >= P1_MIN)).select("s1", "rid", "label")
    extra = [acc.filter(pl.col("label") == 0).with_columns((pl.col("rid").cast(pl.Int64) + (4_000_000_000 - 20_000_000 * i)).cast(pl.UInt32).alias("rid"))
             for i in range(1, dup)]
    pred = pl.concat([acc] + extra).select("s1", "rid")
    return macro_f05(pred, gt_pairs, s1V)


def train():
    best = pl.read_parquet(f"{W}/s2v2_train.parquet")
    base_cols = json.load(open(f"{MD}/model_stage2_cf.json"))["cols"]
    new_cols = base_cols + [c for c in D.FEATURE_NAMES] + NEW_CTX
    best = best.filter(pl.col("p1") >= P1_MIN)
    tr, va = best.filter(~pl.col("isV")), best.filter(pl.col("isV"))
    s1V = pl.read_parquet(f"{W}/val_s1.parquet")
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").rename({"true_s1": "s1"}).join(s1V, on="s1").select("s1", "rid")
    ratio = float(os.environ.get("NEG_RATIO", "2.0"))
    log("train rows", tr.height, "pos", tr["label"].sum(), "| V rows", va.height, "| neg weight", ratio)
    res = {}
    for name, cols in (("base", base_cols), ("new", new_cols)):
        for w in (1.0, ratio):
            wt = np.where(tr["label"].to_numpy() == 1, 1.0, w)
            es_mask = (tr["rid"].hash(seed=3) % 10 == 0).to_numpy()
            X = tr.select(cols).to_numpy().astype(np.float32); y = tr["label"].to_numpy()
            dtr = lgb.Dataset(X[~es_mask], label=y[~es_mask], weight=wt[~es_mask], feature_name=cols, free_raw_data=True)
            dva = lgb.Dataset(X[es_mask], label=y[es_mask], weight=wt[es_mask], reference=dtr)
            params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1)
            m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            del X, dtr, dva; gc.collect()
            p = m.predict(va.select(cols).to_numpy().astype(np.float32), num_threads=2)
            v = va.select("rid", "s1", "p1", "label").with_columns(pl.Series("p2", p))
            rows = []
            for thr in np.arange(0.3, 0.96, 0.05):
                rows.append((float(thr), testlike_f05(v, "p2", thr, gt, s1V, 1), testlike_f05(v, "p2", thr, gt, s1V, int(round(ratio)))))
            best_tr = max(rows, key=lambda r: r[1]); best_tl = max(rows, key=lambda r: r[2])
            key = f"{name}_w{w:g}"
            ev_acc = ev_select(v.filter(pl.col("p1") >= P1_MIN), "p2")
            ev_f = (testlike_f05_pred(ev_acc, v, gt, s1V, 1), testlike_f05_pred(ev_acc, v, gt, s1V, int(round(ratio))))
            res[key] = dict(best_trainlike=best_tr, best_testlike=best_tl, ev=ev_f, iters=m.best_iteration)
            log(key, "train-like best (thr, F)", best_tr[:2], "| test-like best (thr, F)", (best_tl[0], best_tl[2]),
                "| EV-select (train-like, test-like)", ev_f)
            m.save_model(f"{W}/s2v2_{key}.txt")
            v.write_parquet(f"{W}/s2v2_val_{key}.parquet")
            if name == "new" and w == ratio:
                imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
                print([(c, int(g)) for c, g in imp[:30]], flush=True)
    json.dump({"res": res, "base_cols": base_cols, "new_cols": new_cols, "ratio": ratio}, open(f"{W}/s2v2_results.json", "w"), indent=1)


def _dup_negatives(acc_lab: pl.DataFrame, ratio: float) -> pl.DataFrame:
    """copies of accepted negatives so that each counts `ratio` times on average (fractional part by hash)."""
    neg = acc_lab.filter(pl.col("label") == 0)
    full, frac = int(ratio), ratio - int(ratio)
    extra = []
    for i in range(1, full):
        extra.append(neg.with_columns((pl.col("rid").cast(pl.Int64) + (4_000_000_000 - 20_000_000 * i)).cast(pl.UInt32).alias("rid")))
    if frac > 0:
        sel = neg.filter((pl.col("rid").hash(seed=99) % 1000) < int(frac * 1000))
        extra.append(sel.with_columns((pl.col("rid").cast(pl.Int64) + (4_000_000_000 - 20_000_000 * full)).cast(pl.UInt32).alias("rid")))
    return pl.concat([acc_lab] + extra)


def train_cf():
    """Cross-fit T<->V with test-like negative weight, evaluate pooled, then fit the final model on T u V."""
    best = pl.read_parquet(f"{W}/s2v2_train.parquet")
    if "dw_max" not in best.columns:   # self-supervised decoy-word scores on the training table (per country)
        q = pl.read_parquet(f"{W}/train_q.parquet", columns=["rid", "ncore", "country"]).join(best.select("rid").unique(), on="rid")
        s1t = pl.read_parquet(f"{W}/train_s1.parquet", columns=["rid", "ncore"])
        best = add_dw_features(best.drop("country", strict=False).join(q.select("rid", "country"), on="rid", how="left"), q, s1t)
        best.write_parquet(f"{W}/s2v2_train.parquet")
    best = best.filter(pl.col("p1") >= P1_MIN)
    base_cols = json.load(open(f"{MD}/model_stage2_cf.json"))["cols"]
    cols = base_cols + [c for c in D.FEATURE_NAMES] + NEW_CTX + DW_COLS
    ratio = float(os.environ.get("NEG_RATIO", "2.6"))
    s1V = pl.read_parquet(f"{W}/val_s1.parquet"); s1T = pl.read_parquet(f"{W}/T_s1.parquet")
    gt_all = pl.read_parquet(f"{W}/train_gt_pairs.parquet").rename({"true_s1": "s1"})
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1)

    def fit(df, rounds=None):
        wt = np.where(df["label"].to_numpy() == 1, 1.0, ratio)
        X = df.select(cols).to_numpy().astype(np.float32); y = df["label"].to_numpy()
        if rounds is None:
            es = (df["rid"].hash(seed=3) % 10 == 0).to_numpy()
            dtr = lgb.Dataset(X[~es], label=y[~es], weight=wt[~es], feature_name=cols, free_raw_data=True)
            dva = lgb.Dataset(X[es], label=y[es], weight=wt[es], reference=dtr)
            return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        return lgb.train(params, lgb.Dataset(X, label=y, weight=wt, feature_name=cols), num_boost_round=rounds)

    preds, iters = [], []
    for name, tr_mask in (("T->V", ~pl.col("isV")), ("V->T", pl.col("isV"))):
        tr = best.filter(tr_mask); te = best.filter(~tr_mask)
        m = fit(tr); iters.append(m.best_iteration)
        preds.append(te.select("rid", "s1", "p1", "label").with_columns(
            pl.Series("p2", m.predict(te.select(cols).to_numpy().astype(np.float32), num_threads=2))))
        log(name, "iters", m.best_iteration)
    pv = pl.concat(preds)
    s1all = pl.concat([s1V, s1T]).unique()
    gt = gt_all.join(s1all, on="s1").select("s1", "rid")
    res = {}
    acc = ev_select(pv, "p2")
    lab = acc.join(pv.select("rid", "s1", "label"), on=["rid", "s1"])
    res["ev"] = (macro_f05(lab.select("s1", "rid"), gt, s1all), macro_f05(_dup_negatives(lab, ratio).select("s1", "rid"), gt, s1all))
    for thr in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        a = pv.filter(pl.col("p2") >= thr).select("rid", "s1", "label")
        res[f"thr{thr}"] = (macro_f05(a.select("s1", "rid"), gt, s1all), macro_f05(_dup_negatives(a, ratio).select("s1", "rid"), gt, s1all))
    for k, v in res.items():
        log("pooled T+V", k, "train-like %.5f test-like %.5f" % v)
    rounds = int(np.mean(iters) * 1.1)
    m = fit(best, rounds)
    m.save_model(f"{W}/s2v2_cf_final.txt")
    mp = f"{W}/s2v2_results.json"
    meta = json.load(open(mp)) if os.path.exists(mp) else {"res": {}, "base_cols": base_cols, "new_cols": cols, "ratio": ratio}
    meta["cf_cols"] = cols
    meta["train_pos_per_s1"] = float(best["label"].sum() / pl.concat([s1V, s1T]).unique().height)
    meta["train_neg_per_s1"] = float((best.height - best["label"].sum()) / pl.concat([s1V, s1T]).unique().height)
    meta["res"]["cf_final"] = {"pooled": res, "rounds": rounds, "ratio": ratio, "best_testlike": [0.8, None, None]}
    json.dump(meta, open(f"{W}/s2v2_results.json", "w"), indent=1)
    log("final model on T+V saved, rounds", rounds)


def predict():
    cols, models = stage1_models()
    meta = json.load(open(f"{W}/s2v2_results.json"))
    key = os.environ.get("S2KEY", f"new_w{meta['ratio']:g}")
    thr = float(os.environ.get("THR", meta["res"][key]["best_testlike"][0]))
    m2 = lgb.Booster(model_file=f"{W}/s2v2_{key}.txt")
    use_cols = meta["base_cols"] if key.startswith("base") else meta["new_cols"]
    if key == "cf_final" and "cf_cols" in meta:
        use_cols = meta["cf_cols"]
    os.makedirs(OUT, exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet")
    s1dups = stage2.s1_duplicates(s1.select("rid", "country", "ncore", "atoks"))
    vocab = D.s1_vocab(s1)
    pred_path = f"{W}/pred_test_cf.parquet"
    if not os.path.exists(pred_path):
        res = []
        for p in sorted(glob.glob(f"{W}/feat/test_*_part*.parquet")):
            d = read_part(p)
            p1 = np.mean([mm.predict(d.select(cols).to_numpy().astype(np.float32), num_threads=2) for mm in models], axis=0)
            res.append(d.select("rid", "s1").with_columns(pl.Series("p1", p1).cast(pl.Float32)))
        pl.concat(res).write_parquet(pred_path)
        log("stage1 test scored")
    outs = []
    for c in s1["country"].unique().to_list():
        q = pl.scan_parquet(f"{W}/test_q.parquet").filter(pl.col("country") == c).select(QCOLS).collect()
        pr = pl.scan_parquet(pred_path).join(q.select("rid").lazy(), on="rid").collect()
        cf = pl.concat([read_part(p).select("rid", "s1", "a_tset", "nf_tset") for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet"))])
        b = stage2.build(pr, q.select("rid", "entity_id", "ncore", "atoks", "anums"), cf, s1dups)
        del cf, pr; gc.collect()
        b = b.filter(pl.col("p1") >= P1_MIN)
        b = add_new_features(b, q, s1, vocab)
        if any(c_ in use_cols for c_ in DW_COLS):
            b = add_dw_features(b.with_columns(pl.lit(c).alias("country")), q, s1)
        for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet")):
            f = b.join(read_part(p).drop("p1", strict=False), on=["rid", "s1"], how="inner")
            if f.height:
                outs.append(f.select("rid", "s1", "p1").with_columns(
                    pl.Series("p2", m2.predict(f.select(use_cols).to_numpy().astype(np.float32), num_threads=2)).cast(pl.Float32)))
        log("predicted", c)
        del b, q; gc.collect()
    best = pl.concat(outs)
    if os.environ.get("PRIOR_ADJ", "0") == "1":
        # per-country label-shift correction: the model is calibrated for `ratio` x the train negatives per S1;
        # each country's own negatives per S1 are estimated as (candidates per S1 - train positives per S1)
        cc = s1.select(pl.col("rid").alias("s1"), "country")
        nS1 = cc.group_by("country").len().rename({"len": "n"})
        cnt = best.join(cc, on="s1").group_by("country").len().rename({"len": "cand"}).join(nS1, on="country")
        pos, neg, r0 = meta.get("train_pos_per_s1", 3.394), meta.get("train_neg_per_s1", 0.224), float(meta.get("ratio", 2.6))
        if key == "cf_final":
            r0 = float(meta["res"]["cf_final"]["ratio"])
        cnt = cnt.with_columns(((pl.col("cand") / pl.col("n") - pos) / neg).alias("ratio_c")).with_columns((r0 / pl.col("ratio_c")).alias("k"))
        log("prior adjustment per country", cnt.select("country", "ratio_c", "k").to_dicts())
        best = (best.join(cc, on="s1").join(cnt.select("country", "k"), on="country")
                    .with_columns(pl.col("p2").alias("p2_raw"),
                                  (pl.col("p2") * pl.col("k") / (pl.col("p2") * pl.col("k") + 1 - pl.col("p2"))).alias("p2"))
                    .drop("country", "k"))
    best.write_parquet(f"{W}/test_best_s2v2_{key}.parquet")
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
    if os.environ.get("SELECT", "thr") == "ev":
        acc = ev_select(best, "p2").join(qid, on="rid")
    else:
        acc = best.filter(pl.col("p2") >= thr).select("rid", "s1").join(qid, on="rid")
    match = (s1id.join(acc.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
                 .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    cand = best.select("rid", "s1").join(qid, on="rid")
    cp = (s1id.join(cand.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("candidate_entity_ids")), on="s1", how="left")
              .select("source1_entity_id", pl.col("candidate_entity_ids").fill_null("")))
    cp.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    log("written", OUT, "thr", thr, "matched pairs", acc.height)


if __name__ == "__main__":
    {"prep_train": prep_train, "train": train, "train_cf": train_cf, "predict": predict}[STEP]()
