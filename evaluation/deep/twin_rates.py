"""How often does a record with an EMPTY address carry a name that 2+ Source-1 entities share?
Those records are coin flips (nothing in the data tells which twin owns them) and are the largest loss.
Result 2026-09-27 (per 1000 Source-1 entities, exact normalised full name):
  train US 24.3, India 20.6 | test US 18.2, India 20.1, France 33.2  -> France has ~1.7x more,
  which explains about half of France's lower score.
usage: python twin_rates.py <work_dir>"""
import sys

import polars as pl

W = sys.argv[1]
pl.Config.set_tbl_width_chars(200)
for split in ("train", "test"):
    q = pl.read_parquet(f"{W}/{split}_q.parquet", columns=["rid", "business_address", "country", "nfull", "ncore"])
    s = pl.read_parquet(f"{W}/{split}_s1.parquet", columns=["rid", "country", "nfull", "ncore"])
    ns = s.group_by("country").len().rename({"len": "n_S1"})
    sf = s.group_by(["country", "nfull"]).len().rename({"len": "n_same_full"})
    sc = s.group_by(["country", "ncore"]).len().rename({"len": "n_same_core"})
    ea = q.filter(pl.col("business_address").is_null() | (pl.col("business_address").str.strip_chars() == ""))
    x = ea.join(sf, on=["country", "nfull"], how="left").join(sc, on=["country", "ncore"], how="left").fill_null(0)
    r = (x.group_by("country").agg(pl.len().alias("empty_addr_records"), (pl.col("n_same_full") >= 2).sum().alias("with_full_name_twin"),
                                   (pl.col("n_same_core") >= 2).sum().alias("with_core_name_twin"))
          .join(ns, on="country").join(q.group_by("country").len().rename({"len": "records"}), on="country"))
    r = r.with_columns((pl.col("empty_addr_records") / pl.col("records") * 100).round(2).alias("empty_addr_%"),
                       (pl.col("with_full_name_twin") / pl.col("n_S1") * 1000).round(1).alias("full_twin_per_1000_S1"),
                       (pl.col("with_core_name_twin") / pl.col("n_S1") * 1000).round(1).alias("core_twin_per_1000_S1"))
    print(split); print(r.sort("country"))
