"""Run blocking for a split (train|test).
Writes work/{split}_s1.parquet, {split}_q.parquet (all S2+S3 with global row id) and {split}_cand.parquet."""
import sys, time, gc, os
import polars as pl
from blocking import build_s1_index, query

split, W = sys.argv[1], sys.argv[2]
TOPK, REL = 50, 0.2
COLS = ["rid", "country", "ncore", "nalt", "atoks", "pin"]
if not os.path.exists(f"{W}/{split}_q.parquet"):
    pl.scan_parquet(f"{W}/{split}_source1_norm.parquet").with_row_index("rid").sink_parquet(f"{W}/{split}_s1.parquet")
    pl.concat([pl.scan_parquet(f"{W}/{split}_source{i}_norm.parquet") for i in (2, 3)]).with_row_index("rid").sink_parquet(f"{W}/{split}_q.parquet")
countries = pl.scan_parquet(f"{W}/{split}_s1.parquet").select(pl.col("country").unique()).collect()["country"].to_list()
for country in countries:
    if os.path.exists(f"{W}/{split}_cand_{country}.parquet"): continue
    pl.scan_parquet(f"{W}/{split}_q.parquet").filter(pl.col("country") == country).select(COLS).sink_parquet(f"{W}/_tmp_q_{country}.parquet", row_group_size=100000)
parts = []
for country in countries:
    t = time.time()
    p = f"{W}/{split}_cand_{country}.parquet"
    if os.path.exists(p):
        parts.append(p); continue
    s1c = pl.scan_parquet(f"{W}/{split}_s1.parquet").filter(pl.col("country") == country).select(COLS).collect()
    idx = build_s1_index(s1c)
    ns1 = s1c.height
    del s1c; gc.collect()
    sc = query(pl.scan_parquet(f"{W}/_tmp_q_{country}.parquet"), idx, topk=TOPK, rel=REL, chunk=40000)
    p = f"{W}/{split}_cand_{country}.parquet"
    sc.write_parquet(p); parts.append(p)
    print(split, country, "s1", ns1, "pairs", sc.height, "queries", sc["rid"].n_unique(), f"{time.time()-t:.0f}s", flush=True)
    del idx, sc; gc.collect()
pl.concat([pl.scan_parquet(p) for p in parts]).sink_parquet(f"{W}/{split}_cand.parquet")
for country in countries:
    if os.path.exists(f"{W}/_tmp_q_{country}.parquet"): os.remove(f"{W}/_tmp_q_{country}.parquet")
