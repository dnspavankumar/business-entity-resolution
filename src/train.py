"""Stage-1: train the pairwise LightGBM matcher and tune the decision threshold on the held-out split.

usage: python train.py <work_dir> [rounds]
Training queries exclude every query that has a candidate in the validation (V) or
stage-2 training (T) Source-1 subsets, so stage-1 scores on those sets are out-of-sample.
"""
import sys, glob, json, time
import numpy as np
import polars as pl
import lightgbm as lgb
from metrics import macro_f05
from features import add_rank_feats, BREL_MIN

W = sys.argv[1]
DROP = {"rid", "s1", "label", "true_s1", "brank", "p"}


def read_part(p, cols=None):
    d = pl.read_parquet(p)
    d = d.filter(pl.col("bscore") >= BREL_MIN * pl.col("btop"))
    if "brank_min" not in d.columns:
        d = add_rank_feats(d)
    return d if cols is None else d.select(cols + [c for c in ("rid", "s1", "label") if c in d.columns])


def load_np(paths, cols, excl=None):
    ds = []
    for p in paths:
        d = read_part(p, cols)
        if excl is not None:
            d = d.join(excl, on="rid", how="anti")
        ds.append(d)
    n = sum(d.height for d in ds)
    X = np.empty((n, len(cols)), dtype=np.float32); y = np.empty(n, dtype=np.int8)
    i = 0
    for d in ds:
        k = d.height
        X[i:i + k] = d.select(cols).to_numpy().astype(np.float32); y[i:i + k] = d["label"].to_numpy(); i += k
    return X, y


def feat_cols(df):
    return [c for c in df.columns if c not in DROP and not c.startswith("_")]


def eval_thresholds(val, s1v, gt, thrs):
    best = val.sort("p", descending=True).group_by("rid").first().join(s1v, on="s1")
    res = []
    for thr in thrs:
        pred = best.filter(pl.col("p") >= thr).select("s1", "rid")
        res.append((macro_f05(pred, gt, s1v), float(thr)))
    return res


if __name__ == "__main__":
    t0 = time.time()
    parts_trn = sorted(glob.glob(f"{W}/feat/train_trn_part*.parquet"))
    cols = feat_cols(read_part(parts_trn[0]))
    excl = pl.read_parquet(f"{W}/qT.parquet") if len(glob.glob(f"{W}/qT.parquet")) else None
    X, y = load_np(parts_trn, cols, excl)
    parts_val = sorted(glob.glob(f"{W}/feat/train_val_part*.parquet"))
    Xe, ye = load_np(parts_val[::12], cols)
    print("train pairs", len(y), "pos", int(y.sum()), "es pairs", len(ye), "features", len(cols), flush=True)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, max_bin=255)
    dtr = lgb.Dataset(X, label=y, feature_name=cols, free_raw_data=True).construct()
    del X
    dva = lgb.Dataset(Xe, label=ye, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=int(sys.argv[2]) if len(sys.argv) > 2 else 700, valid_sets=[dva],
                  callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)])
    m.save_model(f"{W}/model_stage1.txt")
    print("trained", f"{time.time()-t0:.0f}s", flush=True)
    del dtr, dva
    preds = []
    for p in parts_val:
        v = read_part(p)
        preds.append(v.select("rid", "s1", "label").with_columns(pl.Series("p", m.predict(v.select(cols).to_numpy().astype(np.float32)))))
    val = pl.concat(preds)
    val.write_parquet(f"{W}/val_pred_stage1.parquet")
    print("predicted val", f"{time.time()-t0:.0f}s", flush=True)
    s1v = pl.read_parquet(f"{W}/val_s1.parquet")
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").rename({"true_s1": "s1"}).join(s1v, on="s1")
    res = eval_thresholds(val, s1v, gt, np.arange(0.3, 0.96, 0.05))
    for f, thr in res:
        print(f"thr={thr:.2f} F0.5={f:.5f}")
    best = max(res)
    print("BEST", best)
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
    print([(c, int(g)) for c, g in imp[:40]])
    json.dump({"cols": cols, "thr": best[1], "val_f05": best[0]}, open(f"{W}/model_stage1.json", "w"))
    print(f"{time.time()-t0:.0f}s")
