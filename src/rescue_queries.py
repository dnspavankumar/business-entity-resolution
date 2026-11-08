"""Query lists for the rescue step.

1) Records that the main blocking gave no candidate at all (train and test):
   <work>/rescue_q_train.parquet, <work>/rescue_q_test.parquet   (column rid)
2) France records whose best Source-1 candidate has stage-1 p1 < 0.02 (they never reach stage 2) and
   that differ from that entity only by one noise suffix (Services / France / Developpement / Group /
   Co / & Associes / & Fils) at the same or nearly the same address. This class has the true-match
   per-entity profile (see france_fix2.py). -> <work>/fr_lowp1_noise_pairs.parquet (rid, s1)

usage: python rescue_queries.py <work_dir>
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoy_features as D  # noqa: E402

NOISE = ["services", "service", "developpement", "france", "group", "co", "associes", "fils"]


def main():
    W = sys.argv[1]
    for split in ("train", "test"):
        c = pl.scan_parquet(f"{W}/{split}_cand.parquet").select("rid").unique().collect()
        q = pl.read_parquet(f"{W}/{split}_q.parquet", columns=["rid"])
        nc = q.join(c, on="rid", how="anti")
        nc.write_parquet(f"{W}/rescue_q_{split}.parquet")
        print(split, "records without candidates", nc.height, flush=True)
    # France low-p1 noise-suffix pairs
    qf = pl.read_parquet(f"{W}/test_q.parquet").filter(pl.col("country") == "France")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet")
    best = (pl.scan_parquet(f"{W}/pred_test_cf.parquet").join(qf.select("rid").lazy(), on="rid", how="semi")
              .sort("p1", descending=True).group_by("rid").first().collect())
    low = best.filter(pl.col("p1") < 0.02).select("rid", "s1")
    vocab = D.s1_vocab(s1)
    f = D.compute(low, qf, s1, vocab).select("rid", "s1", "d1", "street_jac", "city_overlap", "legal_added")
    ops = D.op_tokens(low, qf, s1)
    f = f.join(ops, on=["rid", "s1"]).with_columns(pl.col("added").list.len().alias("na"), pl.col("dropped").list.len().alias("nd"),
                                                    pl.col("added").list.first().alias("w"))
    near = (pl.col("d1") == 0) & (pl.col("street_jac") >= 0.3) & (pl.col("city_overlap") == 1)
    rule = near & (pl.col("na") == 1) & (pl.col("nd") <= 1) & pl.col("w").is_in(NOISE) & (pl.col("legal_added") != 1)
    out = f.filter(rule.fill_null(False)).select("rid", "s1")
    out.write_parquet(f"{W}/fr_lowp1_noise_pairs.parquet")
    print("France low-p1 noise-suffix pairs", out.height, flush=True)


if __name__ == "__main__":
    main()
