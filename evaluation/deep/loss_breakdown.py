"""Where is the remaining local loss? Oracle gain of fixing each bucket (US+India, labelled V+T entities).
Buckets (from <work_dir>/reach_true.parquet): a = record has no candidate, b = true Source-1 not among its candidates,
c = true Source-1 not ranked first, d = ranked first but p1 < 0.02, e = reachable but rejected; each split by
empty address (EA) or not; plus "remove all false positives".
Result 2026-09-27 for v17 (local 0.98955 before the rescue): EA buckets together ~0.006, a[addr] 0.0016,
b[addr] 0.0007, c[addr] 0.0005, e[addr] 0.0007, all false positives 0.0010.
usage: python loss_breakdown.py <work_dir> <preds.parquet> [score_column, default p2x] [K scale, default 1.5]
       preds = rid, s1, <score> for the V+T best pairs (e.g. work/v17_local_preds.parquet)"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from local_eval import LocalEval, ev_select, K_V10, RATIO, TEST_S1  # noqa: E402

W, P = sys.argv[1], sys.argv[2]
COL = sys.argv[3] if len(sys.argv) > 3 else "p2x"
KS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
le = LocalEval(W)
pv = pl.read_parquet(P).select("rid", "s1", pl.col(COL).alias("p"))
b = pv.join(le.s1.select("s1", "country"), on="s1")
reach = pl.read_parquet(f"{W}/reach_true.parquet").select("rid", "s1", "case", "ea")
res = {}
for c in ("US", "India"):
    s1c = le.s1.filter(pl.col("country") == c)
    k = K_V10[c] * KS
    p = b.filter(pl.col("country") == c).with_columns((pl.col("p") * k / (pl.col("p") * k + 1 - pl.col("p"))).alias("_p"))
    acc = ev_select(p, "_p").join(s1c.select("s1"), on="s1")
    gt = le.gt.join(s1c.select("s1"), on="s1").with_columns(pl.lit(1).alias("t"))

    def score(pred):
        lab = (pred.join(gt, on=["s1", "rid"], how="left").with_columns(pl.col("t").fill_null(0))
                   .join(le.matched_rids, on="rid", how="left").with_columns(pl.col("is_match_rec").fill_null(False)))
        w = pl.when(pl.col("t") == 1).then(1.0).when(pl.col("is_match_rec")).then(1.0).otherwise(RATIO[c])
        return le._macro(lab.with_columns(w.alias("w")).select("s1", "rid", "w"), s1c)
    base = score(acc)
    lab = acc.join(gt, on=["s1", "rid"], how="left").with_columns(pl.col("t").fill_null(0))
    tp = lab.filter(pl.col("t") == 1).select("rid", "s1")
    missed = gt.select("rid", "s1").join(acc, on=["rid", "s1"], how="anti").join(reach.join(s1c.select("s1"), on="s1"), on=["rid", "s1"], how="left")
    out = {"base": (base, 0)}
    for case in ("a no candidates at all", "b true S1 not in candidates", "c true S1 not ranked first", "d first but p1<0.02", "e reachable"):
        for ea in (False, True):
            add = missed.filter((pl.col("case") == case) & (pl.col("ea") == ea)).select("rid", "s1")
            out[f"{case} [{'empty address' if ea else 'address'}]"] = (score(pl.concat([acc, add])) - base, add.height)
    out["remove all false positives"] = (score(tp) - base, lab.height - tp.height)
    res[c] = out
wt = {c: TEST_S1[c] / sum(TEST_S1.values()) for c in TEST_S1}
print("base", {c: round(res[c]["base"][0], 5) for c in res})
for kk in res["US"]:
    if kk != "base":
        print(f"  {kk:48s} oracle gain {sum(wt[c] * res[c][kk][0] for c in wt):+.5f}   pairs {sum(res[c][kk][1] for c in wt)}")
