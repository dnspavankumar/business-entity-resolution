"""Pick an extra stage-2 training subset (default T2 = 10% of train Source-1 entities, hash 7..16, disjoint from
V (hash 0..2) and T (3..6)) and the queries that have a candidate in it.
usage: python select_T2.py <work_dir> [name lo hi]      e.g. select_T2.py work T3 17 27"""
import sys
import polars as pl

W = sys.argv[1]
NAME, LO, HI = (sys.argv[2], int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else ("T2", 7, 17)
h = pl.col("entity_id").hash(seed=42) % 100
s1t = pl.scan_parquet(f"{W}/train_s1.parquet").select("rid", "entity_id").filter((h >= LO) & (h < HI)).select(pl.col("rid").alias("s1")).collect()
s1t.write_parquet(f"{W}/{NAME}_s1.parquet")
qt = (pl.scan_parquet(f"{W}/train_cand.parquet").select("rid", "s1", "bscore", "btop").filter(pl.col("bscore") >= 0.3 * pl.col("btop"))
      .join(s1t.lazy(), on="s1").select("rid").unique().collect(engine="streaming"))
qt.write_parquet(f"{W}/q{NAME}.parquet")
print(NAME, "s1", s1t.height, "queries", qt.height)
