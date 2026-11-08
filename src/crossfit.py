"""Full cross-fitting on the training data.

1. every train query gets exactly one "home" feature part (priority val > T > trn > rest)
2. queries are split into 2 folds by hash; stage-1 model k is trained on fold k (queries of the
   validation subset V are never used for training) and scores fold 1-k  -> out-of-fold p1 for ALL pairs
3. stage-2 context is therefore complete for every Source-1 entity; stage 2 is trained on a large
   sample of best pairs from queries outside V and evaluated on V.

usage: python crossfit.py <work_dir> [step]   step in {s1, s2, all}
"""
import sys, glob, json, time, gc, os
import numpy as np
import polars as pl
import lightgbm as lgb
import stage2
from train import read_part, feat_cols, DROP
from metrics import macro_f05

W = sys.argv[1]
STEP = sys.argv[2] if len(sys.argv) > 2 else "all"
t0 = time.time()
PRIO = ["train_val", "train_T", "train_trn", "train_rest"]
MAX_PAIRS = 10_500_000
log = lambda *a: print(*a, f"{time.time()-t0:.0f}s", flush=True)


def parts_of(prefix):
    if prefix == "train_rest":
        return sorted(glob.glob(f"{W}/feat/train_rest*_part*.parquet"))
    return sorted(glob.glob(f"{W}/feat/{prefix}_part*.parquet"))


def home_table():
    p = f"{W}/cf_home.parquet"
    if os.path.exists(p):
        return pl.read_parquet(p)
    seen = None
    out = []
    for pre in PRIO:
        r = pl.concat([pl.scan_parquet(x).select("rid") for x in parts_of(pre)]).unique().collect()
        if seen is not None:
            r = r.join(seen, on="rid", how="anti")
        out.append(r.with_columns(pl.lit(pre).alias("home")))
        seen = r if seen is None else pl.concat([seen, r])
    h = pl.concat(out).with_columns((pl.col("rid").hash(seed=11) % 2).cast(pl.Int8).alias("fold"))
    h.write_parquet(p)
    return h


def iter_rows(home, cols_needed=None):
    """yield (prefix, df) for every part, restricted to rows whose query lives in that part-set."""
    for pre in PRIO:
        hh = home.filter(pl.col("home") == pre).select("rid", "fold")
        for p in parts_of(pre):
            d = read_part(p)
            d = d.join(hh, on="rid")
            yield pre, (d if cols_needed is None else d.select(cols_needed))


def stage1(home):
    qV = home.filter(pl.col("home") == "train_val").select("rid")
    cols = feat_cols(read_part(parts_of("train_trn")[0]))
    cols = [c for c in cols if c != "fold"]
    params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, max_bin=255)
    models = []
    for k in (0, 1):
        # training queries of fold k (outside V), sampled to fit memory
        cand_q = home.filter((pl.col("fold") == k) & (pl.col("home") != "train_val"))
        frac = 0.62
        keep = cand_q.sample(fraction=frac, seed=k).select("rid")
        Xs, ys = [], []
        n = 0
        for pre, d in iter_rows(home):
            d = d.join(keep, on="rid")
            if d.height == 0:
                continue
            Xs.append(d.select(cols).to_numpy().astype(np.float32)); ys.append(d["label"].to_numpy().astype(np.int8)); n += d.height
            if n > MAX_PAIRS:
                break
        X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
        log(f"fold {k}: train pairs {len(y)} pos {int(y.sum())}")
        # small early-stopping set: V pairs
        es = pl.concat([read_part(p).select(cols + ["label"]) for p in parts_of("train_val")[::15]])
        dtr = lgb.Dataset(X, label=y, feature_name=cols, free_raw_data=True).construct(); del X
        dva = lgb.Dataset(es.select(cols).to_numpy().astype(np.float32), label=es["label"].to_numpy(), reference=dtr); del es
        m = lgb.train(params, dtr, num_boost_round=900, valid_sets=[dva], callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)])
        m.save_model(f"{W}/model_stage1_f{k}.txt")
        models.append(m)
        del dtr, dva; gc.collect()
        log(f"fold {k} trained, iters {m.best_iteration}")
    json.dump({"cols": cols}, open(f"{W}/model_stage1_cf.json", "w"))
    # out-of-fold scores for every pair
    res = []
    for pre, d in iter_rows(home):
        p = np.empty(d.height, dtype=np.float32)
        for k in (0, 1):
            msk = (d["fold"] == (1 - k)).to_numpy()   # scored by the model that did NOT train on this fold
            if msk.any():
                p[msk] = models[k].predict(d.filter(pl.Series(msk)).select(cols).to_numpy().astype(np.float32), num_threads=2)
        res.append(d.select("rid", "s1", "label", "a_tset", "nf_tset").with_columns(pl.Series("p1", p)))
    oof = pl.concat(res)
    oof.write_parquet(f"{W}/oof_p1.parquet")
    log("oof written", oof.height)


def stage2_train(home):
    s1 = pl.read_parquet(f"{W}/train_s1.parquet", columns=["rid", "country", "ncore", "atoks"])
    s1dups = stage2.s1_duplicates(s1); del s1
    qV = home.filter(pl.col("home") == "train_val").select("rid")
    s1V = pl.read_parquet(f"{W}/val_s1.parquet")
    for c in ("US", "India"):
        pth = f"{W}/cf_best_{c}.parquet"
        if os.path.exists(pth):
            continue
        qtext = pl.scan_parquet(f"{W}/train_q.parquet").filter(pl.col("country") == c).select("rid", "entity_id", "ncore", "atoks", "anums").collect()
        o = pl.scan_parquet(f"{W}/oof_p1.parquet").join(qtext.select("rid").lazy(), on="rid").collect()
        b = stage2.build(o.select("rid", "s1", "p1"), qtext, o.select("rid", "s1", "a_tset", "nf_tset"), s1dups)
        b.write_parquet(pth)
        del qtext, o, b; gc.collect()
        log("context", c)
    best = pl.concat([pl.read_parquet(f"{W}/cf_best_{c}.parquet") for c in ("US", "India")], how="diagonal_relaxed")
    # rows used: V rows (evaluation, s1 in V) + a sample of non-V queries for training
    Vkeys = best.join(qV, on="rid").join(s1V, on="s1").select("rid", "s1")
    Tr = best.join(qV, on="rid", how="anti").filter(pl.col("p1") >= 0.02)
    Tr = Tr.sample(n=min(2_500_000, Tr.height), seed=5).select("rid", "s1")
    keys = pl.concat([Vkeys.with_columns(pl.lit(1).alias("isV")), Tr.with_columns(pl.lit(0).alias("isV"))])
    feats = []
    for pre, d in iter_rows(home):
        feats.append(d.join(keys, on=["rid", "s1"]))
    F = pl.concat(feats, how="diagonal_relaxed"); del feats
    F = F.join(best, on=["rid", "s1"], how="left")
    log("stage2 table", F.height)
    drop = set(DROP) | {"src", "isV", "fold", "true_s1"}
    cols = [c for c in F.columns if c not in drop]
    tr = F.filter(pl.col("isV") == 0); va = F.filter(pl.col("isV") == 1)
    del F; gc.collect()
    # early stopping on a 5% slice of training rows (never on V)
    tr = tr.with_columns((pl.col("rid").hash(seed=3) % 20 == 0).alias("_es"))
    es = tr.filter(pl.col("_es")); tr = tr.filter(~pl.col("_es"))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1)
    dtr = lgb.Dataset(tr.select(cols).to_numpy().astype(np.float32), label=tr["label"].to_numpy(), feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(es.select(cols).to_numpy().astype(np.float32), label=es["label"].to_numpy(), reference=dtr)
    del tr, es; gc.collect()
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)])
    m.save_model(f"{W}/model_stage2_cf.txt")
    log("stage2 trained", m.best_iteration)
    va = va.with_columns(pl.Series("p2", m.predict(va.select(cols).to_numpy().astype(np.float32))))
    va.select("rid", "s1", "p1", "p2", "label").write_parquet(f"{W}/val_pred_stage2_cf.parquet")
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").rename({"true_s1": "s1"}).join(s1V, on="s1")
    best_thr = (0, 0)
    for thr in np.arange(0.5, 0.925, 0.025):
        f = macro_f05(va.filter((pl.col("p2") >= thr) & (pl.col("p1") >= 0.02)).select("s1", "rid"), gt, s1V)
        print(f"thr={thr:.3f} F0.5={f:.5f}")
        best_thr = max(best_thr, (f, float(thr)))
    f1 = max(macro_f05(va.filter(pl.col("p1") >= t).select("s1", "rid"), gt, s1V) for t in (0.6, 0.65, 0.7, 0.75))
    log("BEST stage2", best_thr, "stage1-only (oof)", f1)
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
    print([(c, int(g)) for c, g in imp[:25]])
    json.dump({"cols": cols, "thr": best_thr[1], "val_f05": best_thr[0]}, open(f"{W}/model_stage2_cf.json", "w"))


if __name__ == "__main__":
    home = home_table()
    log("home", home.group_by("home").len().to_dicts())
    if STEP in ("s1", "all"):
        stage1(home)
    if STEP in ("s2", "all"):
        stage2_train(home)
