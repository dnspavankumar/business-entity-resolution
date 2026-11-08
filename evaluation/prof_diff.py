"""Label-free check of a new submission against an old one, per country (works for France, which has no labels).

Per-entity count profile: for a class of pairs, count per Source-1 entity how many pairs of the class it has,
as a function of n_ref = the number of OTHER accepted records of that entity (0..6).
  * true noise grows with entity size and singletons have almost none: profile ~ [0.28, 1, 0.82, 0.72, 0.65, 0.60, 0.53]
  * decoys are generated once per entity, so the profile is flat:      profile ~ [1.09, 1, 1.01, 0.97, 0.96, 0.92, 0.86]
A least-squares mix of the two gives the decoy share of the class.
Here the classes are "pairs the new file ADDS" and "pairs it REMOVES".

CAUTION (learnt the hard way): the estimate is biased when the change itself depends on cluster context
(e.g. a new p2 threshold): on labelled US data, pairs added by K x1.5 were 86% true but the profile said 40%.
Use it for rule-based changes of a pair class; confirm on US/India with evaluate.py whenever possible.

usage: python prof_diff.py <work_dir> <old matching_results.tsv> <new matching_results.tsv>
"""
import sys

import numpy as np
import polars as pl

W = sys.argv[1]
T = np.array([0.28, 1, 0.82, 0.72, 0.65, 0.60, 0.53])
D = np.array([1.09, 1, 1.01, 0.97, 0.96, 0.92, 0.86])
s1id = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "country"])
qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])


def pairs(tsv):
    sub = pl.read_csv(tsv, separator="\t", infer_schema=False, quote_char=None)
    return (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
               .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
               .join(s1id.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("rid").alias("s1"), "country"), on="source1_entity_id")
               .join(qid.rename({"entity_id": "m"}), on="m").select("rid", "s1", "country"))


def fit(cls, acc, s1all):
    """cls: pairs of the class (all accepted in `acc`); acc: all accepted pairs of that submission."""
    nacc = acc.group_by("s1").agg(pl.len().alias("n_all"))
    c = cls.group_by("s1").agg(pl.len().alias("cnt"))
    g = (s1all.join(c, on="s1", how="left").join(nacc, on="s1", how="left").fill_null(0)
              .with_columns((pl.col("n_all") - pl.col("cnt")).clip(0, 6).alias("kb")))
    agg = g.group_by("kb").agg(pl.col("cnt").mean().alias("d"), pl.len().alias("S")).sort("kb")
    if agg.height < 7:
        return None, None
    d = agg["d"].to_numpy(); S = agg["S"].to_numpy()
    X = np.stack([T, D], 1); w = np.sqrt(S)
    coef, _, _, _ = np.linalg.lstsq(X * w[:, None], d * w, rcond=None)
    t, e = np.clip(coef, 0, None)
    tot = (d * S).sum(); dec = (e * D * S).sum()
    return np.round(d / max(d[1], 1e-9), 2), dec / max(tot, 1e-9)


old, new = pairs(sys.argv[2]), pairs(sys.argv[3])
for c in ("US", "India", "France"):
    s1all = s1id.filter(pl.col("country") == c).select(pl.col("rid").alias("s1"))
    o, n = old.filter(pl.col("country") == c), new.filter(pl.col("country") == c)
    add = n.join(o, on=["rid", "s1"], how="anti")
    rem = o.join(n, on=["rid", "s1"], how="anti")
    pa, sa = fit(add, n, s1all)
    pr, sr = fit(rem, o, s1all)
    print(f"{c:7s} old {o.height} new {n.height} | ADDED {add.height} decoy share {sa if sa is None else round(sa, 2)} prof {pa}"
          f" | REMOVED {rem.height} decoy share {sr if sr is None else round(sr, 2)} prof {pr}", flush=True)
