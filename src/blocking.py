"""Candidate generation: for every Source-2/3 record retrieve Source-1 records sharing
selective blocking keys, score by summed IDF of shared keys, keep the top candidates."""
import time, sys
import numpy as np
import polars as pl
from keys import make_keys, make_keys_chunked, CAPS, CAPS_NOADDR, S1_MAXCAP


def build_s1_index(s1: pl.DataFrame, chunk: int = 100000):
    """s1 must have rid (u32).  Returns key table (key, s1, w, df, kind)."""
    k1 = make_keys_chunked(s1, chunk)
    df = k1.group_by("key").agg(pl.len().alias("df"), pl.col("kind").first())
    df = df.filter(pl.col("df") <= S1_MAXCAP)
    n = max(s1.height, 1)
    df = df.with_columns((np.log(n) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32).alias("w"))
    k1 = k1.join(df.select("key", "df", "w"), on="key")
    return k1.select("key", pl.col("rid").alias("s1"), "w", "df", "kind")


def query(o: pl.DataFrame, idx: pl.DataFrame, topk: int = 20, chunk: int = 200000, rel: float = 0.0):
    """o must have rid (u32), atoks.  Returns (rid, s1, bscore, nkeys, brank, btop)."""
    out = []
    capx = pl.col("kind").replace_strict(CAPS, return_dtype=pl.Int32)
    capn = pl.col("kind").replace_strict(CAPS_NOADDR, return_dtype=pl.Int32)
    n = o.height if isinstance(o, pl.DataFrame) else o.select(pl.len()).collect().item()
    for i in range(0, n, chunk):
        oc = o.slice(i, chunk)
        if isinstance(oc, pl.LazyFrame):
            oc = oc.collect()
        k2 = make_keys(oc)
        noaddr = oc.filter(pl.col("atoks").str.strip_chars() == "").select("rid").with_columns(pl.lit(True).alias("na"))
        k2 = k2.join(noaddr, on="rid", how="left").with_columns(pl.col("na").fill_null(False))
        j = k2.select("rid", "key", "na").join(idx, on="key")
        j = j.filter(pl.col("df") <= pl.when(pl.col("na")).then(capn).otherwise(capx))
        sc = j.group_by(["rid", "s1"]).agg(pl.col("w").sum().alias("bscore"), pl.len().alias("nkeys"))
        sc = sc.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("rid").alias("brank"),
                             pl.col("bscore").max().over("rid").alias("btop"))
        out.append(sc.filter((pl.col("brank") <= topk) & (pl.col("bscore") >= rel * pl.col("btop")))
                   .with_columns(pl.col("brank").cast(pl.UInt16), pl.col("nkeys").cast(pl.UInt16)))
    return pl.concat(out)
