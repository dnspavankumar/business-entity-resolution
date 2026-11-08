"""Pre-compute the v14 extra features (extra_feats.py) for the test best pairs, one country per call, so that
stage2_v14.py predict only joins them (keeps peak memory low).
usage: python extra_test.py <work_dir> <country>"""
import sys
import time

import polars as pl

import extra_feats as E

W, C = sys.argv[1], sys.argv[2]
t0 = time.time()
q = pl.scan_parquet(f"{W}/test_q.parquet").filter(pl.col("country") == C).select("rid", "entity_id", "anums", "atoks", "acity", "ncore").collect()
best = (pl.scan_parquet(f"{W}/test_best_s2v2_cf_final_v10.parquet").select("rid", "s1", "p1")
          .join(q.select("rid").lazy(), on="rid").filter(pl.col("p1") >= 0.02).collect())
s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "anums", "atoks", "acity", "ncore"]).join(best.select(pl.col("s1").alias("rid")).unique(), on="rid")
ex = E.compute(best, q, s1)
ex.write_parquet(f"{W}/test_extra_{C}.parquet")
print(C, "extra features", ex.shape, f"{time.time()-t0:.0f}s", flush=True)
