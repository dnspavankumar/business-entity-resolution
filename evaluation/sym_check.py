"""Label-free symmetry check of accepted pairs on the TEST set.

True house-number noise is symmetric (a record is +k from its Source-1 as often as -k), while decoys are only
shifted UP by {1,2,3,4,5,7,9,11,13,21}. So for accepted pairs:
    excess = count(d1 = +k) - count(d1 = -k)   ~  number of accepted decoys with offset k.
Compare two submissions: if the new one raises excess_strong, it accepts more decoys (this is how the
v14 sibling-agreement features were caught: +0.0004 locally, but more upward-offset decoys on the test).

needs <work_dir>/test_best_feats_<country>.parquet (column d1 = record number - Source-1 number),
written by stage2_v14.py predict.
usage: python sym_check.py <work_dir> <matching_results.tsv> [more.tsv ...]
"""
import sys

import polars as pl

W = sys.argv[1]
STRONG = [3, 4, 5, 7, 9, 11, 13, 21]
s1id = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "country"])
qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])


def pairs(tsv):
    sub = pl.read_csv(tsv, separator="\t", infer_schema=False, quote_char=None)
    return (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
               .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
               .join(s1id.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("rid").alias("s1"), "country"), on="source1_entity_id")
               .join(qid.select("rid", pl.col("entity_id").alias("m")), on="m").select("rid", "s1", "country"))


for name in sys.argv[2:]:
    acc = pairs(name)
    rows = []
    for c in ("US", "India", "France"):
        f = pl.read_parquet(f"{W}/test_best_feats_{c}.parquet", columns=["rid", "s1", "d1"])
        a = f.join(acc.filter(pl.col("country") == c), on=["rid", "s1"]).filter(pl.col("d1").abs().is_between(1, 21))
        a = a.with_columns(pl.col("d1").cast(pl.Int32))
        pos = a.filter(pl.col("d1").is_in(STRONG)).height
        neg = a.filter((-pl.col("d1")).is_in(STRONG)).height
        p12 = a.filter(pl.col("d1").is_in([1, 2])).height
        n12 = a.filter(pl.col("d1").is_in([-1, -2])).height
        rows.append((c, pos, neg, pos - neg, p12, n12, p12 - n12))
    print(name)
    print(pl.DataFrame(rows, schema=["country", "strong_pos", "strong_neg", "excess_strong", "pm12_pos", "pm12_neg", "excess_12"], orient="row"))
