"""Stage-2 training rows for the extra subset T2 (select_T2.py, build_features_t2.py) — same recipe as
stage2_v2.prep_train for V and T:
  out-of-fold stage-1 scores (fold models in models/), best pair per query, cluster context (stage2.build),
  decoy/noise features, stage-1 feature rows of the best pairs, labels.
Writes <work>/s2v2_train_T2.parquet (only best pairs whose Source-1 entity is in T2, so every cluster is complete).
usage: python prep_T2.py <work_dir> <models_dir> [subset name, default T2]
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
from train import read_part

W, MD = sys.argv[1], sys.argv[2]
SUB = sys.argv[3] if len(sys.argv) > 3 else "T2"
t0 = time.time()
log = lambda *a: print(*a, f"{time.time()-t0:.0f}s", flush=True)
QCOLS = ["rid", "entity_id", "country", "ncore", "nfull", "atoks", "anums", "acity"]


def main():
    meta = json.load(open(f"{MD}/model_stage1_cf.json"))
    cols = meta["cols"]
    models = [lgb.Booster(model_file=f"{MD}/model_stage1_f{k}.txt") for k in (0, 1)]
    parts = sorted(glob.glob(f"{W}/feat/train_{SUB}_part*.parquet"))
    out_dir = f"{W}/s2v2_oof_{SUB}"
    os.makedirs(out_dir, exist_ok=True)
    prev = [f"{W}/s2v2_oof/part*.parquet"] + [f"{d}/part*.parquet" for d in sorted(glob.glob(f"{W}/s2v2_oof_T*")) if not d.endswith(SUB)]
    seen = pl.concat([pl.read_parquet(g, columns=["rid"]) for g in prev]).unique()
    for i, p in enumerate(parts):
        outp = f"{out_dir}/part{i:04d}.parquet"
        if os.path.exists(outp):
            continue
        d = read_part(p).join(seen, on="rid", how="anti")
        if d.height == 0:
            continue
        fold = (d["rid"].hash(seed=11) % 2).to_numpy()
        p1 = np.empty(d.height, dtype=np.float32)
        for k in (0, 1):
            m = fold == (1 - k)
            if m.any():
                p1[m] = models[k].predict(d.filter(pl.Series(m)).select(cols).to_numpy().astype(np.float32), num_threads=2)
        d.select("rid", "s1", "label", "a_tset", "nf_tset").with_columns(pl.Series("p1", p1)).write_parquet(outp)
        del d
        gc.collect()
        if i % 10 == 0:
            log("oof part", i, "of", len(parts))
    s1T2 = pl.read_parquet(f"{W}/{SUB}_s1.parquet")
    qT2 = pl.read_parquet(f"{W}/q{SUB}.parquet")
    oof = pl.concat([pl.scan_parquet(g).join(qT2.lazy(), on="rid", how="semi").collect() for g in prev + [f"{out_dir}/part*.parquet"]])
    log("oof rows for T2 queries", oof.height)
    s1 = pl.read_parquet(f"{W}/train_s1.parquet")
    s1dups = stage2.s1_duplicates(s1.select("rid", "country", "ncore", "atoks"))
    q = pl.read_parquet(f"{W}/train_q.parquet", columns=QCOLS).join(qT2, on="rid")
    vocab = D.s1_vocab(s1)
    outs = []
    for c in q["country"].unique().to_list():
        qc = q.filter(pl.col("country") == c)
        o = oof.join(qc.select("rid"), on="rid")
        b = stage2.build(o.select("rid", "s1", "p1"), qc.select("rid", "entity_id", "ncore", "atoks", "anums"),
                         o.select("rid", "s1", "a_tset", "nf_tset"), s1dups)
        b = b.join(s1T2, on="s1")
        f = D.compute(b.select("rid", "s1"), qc, s1, vocab)
        cl = D.cluster_features(b.select("rid", "s1", "p1"), f.select("rid", "s1", "d1"), qc)
        b = b.join(f, on=["rid", "s1"], how="left").join(cl, on=["rid", "s1"], how="left")
        outs.append(b)
        log("built", c, b.height)
        del o
        gc.collect()
    best = pl.concat(outs, how="diagonal_relaxed")
    del oof
    gc.collect()
    keys = best.select("rid", "s1")
    fr = []
    for p in sorted(glob.glob(f"{W}/feat/train_*_part*.parquet")):
        x = read_part(p).join(keys, on=["rid", "s1"]).drop("label", "true_s1", strict=False)
        if x.height:
            fr.append(x)
    F = pl.concat(fr, how="diagonal_relaxed").unique(["rid", "s1"])
    del fr
    gc.collect()
    best = best.join(F, on=["rid", "s1"], how="left")
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet")
    best = best.join(gt, on="rid", how="left").with_columns((pl.col("true_s1") == pl.col("s1")).fill_null(False).cast(pl.Int8).alias("label"))
    best = best.with_columns(pl.lit(False).alias("isV"), pl.lit(True).alias("isT2"))
    best.write_parquet(f"{W}/s2v2_train_{SUB}.parquet")
    log("saved", best.height, "positives", int(best["label"].sum()))


if __name__ == "__main__":
    main()
