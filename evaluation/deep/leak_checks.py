"""Leak checks on the train ground truth (result 2026-09-27: no leak).
  * does the Source-1 id predict the id of its records? (id mod m equal at chance rate, correlation 0)
  * how many records per source does an entity have? (Source-2: 0-5, Source-3: 0-6, broad spread)
usage: python leak_checks.py <work_dir>        (needs <work_dir>/data/train_ground_truth.parquet from convert.py)"""
import sys

import numpy as np
import polars as pl

W = sys.argv[1]
g = pl.read_parquet(f"{W}/data/train_ground_truth.parquet")
e = (g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
      .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")))
e = e.with_columns(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("a"),
                   pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64).alias("b"),
                   pl.col("matched_entity_ids").str.slice(0, 2).alias("src"))
a, b = e["a"].to_numpy(), e["b"].to_numpy()
bp = np.random.default_rng(0).permutation(b)
print("true pairs", e.height)
for m in (2, 3, 5, 7, 10, 13, 97, 100, 1000):
    print(f"id mod {m:5d}: equal for true pairs {np.mean(a % m == b % m):.4f}   for shuffled pairs {np.mean(a % m == bp % m):.4f}")
print("correlation of ids", round(float(np.corrcoef(a, b)[0, 1]), 5))
c = e.group_by("source1_entity_id").agg((pl.col("src") == "S2").sum().alias("n_source2"), (pl.col("src") == "S3").sum().alias("n_source3"))
print(c.group_by(["n_source2", "n_source3"]).len().sort("len", descending=True).head(12))
