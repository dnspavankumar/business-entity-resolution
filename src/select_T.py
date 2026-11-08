"""Pick the stage-2 training subset T (4% of train Source-1 entities, disjoint from the 3% validation subset V)
and the queries that have a candidate in T.  usage: python select_T.py <work_dir>"""
import sys
import polars as pl

W = sys.argv[1]
h = pl.col("entity_id").hash(seed=42) % 100
s1t = pl.scan_parquet(f"{W}/train_s1.parquet").select("rid", "entity_id").filter((h >= 3) & (h < 7)).select(pl.col("rid").alias("s1")).collect()
s1t.write_parquet(f"{W}/T_s1.parquet")
qt = (pl.scan_parquet(f"{W}/train_cand.parquet").select("rid", "s1", "bscore", "btop").filter(pl.col("bscore") >= 0.3 * pl.col("btop"))
      .join(s1t.lazy(), on="s1").select("rid").unique().collect(engine="streaming"))
qt.write_parquet(f"{W}/qT.parquet")
print("T s1", s1t.height, "T queries", qt.height)
