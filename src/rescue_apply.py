"""Add the rescued pairs (records with no blocking candidate, see rescue_cands.py / rescue_score.py)
to an existing submission.

For every rescued record we take its best rescue candidate. Its probability is moved to the test
prior (odds / ODDS_K, chosen on the local US/India harness: +0.00066 US+India F) and the pair joins
the entity's own pairs. The expected-F0.5 selection is re-run only for the entities that receive a
rescued pair; all other rows are kept from the input file. The entity's own pairs keep the
probabilities used by the input submission: decoy post-filter removals stay removed and, for France,
the france_fix2.py DECOY (-> 0) and NOISE (-> max(p2, 0.97)) corrections are applied again.

France records whose best Source-1 candidate scored p1 < 0.02 in stage 1 never reach stage 2. Among
them, "one noise suffix added at the same (or nearly the same) address" records have the true-match
profile (see france_fix2.py); they are offered to the selection with p = 0.97 (optional 7th argument:
a parquet with rid, s1 of such pairs).

usage: python rescue_apply.py <work_dir> <in_matching_results.tsv> <removed_pairs.tsv>
                              <france_fix2_pairs.parquet> <rescue_pred_test.parquet> <out_dir> [extra_pairs.parquet]
                              [in_candidate_pairs.tsv]
With the last argument, <out_dir>/candidate_pairs.tsv is written: the input candidates plus every rescued
pair offered to the final selection (the rescue search is part of our candidate generation).
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from france_fix import ev_select  # noqa: E402

ODDS_K = 2.0
MIN_P = 0.05
P_NOISE = 0.97


def main():
    W, in_tsv, removed_tsv, ff2, resc, out_dir = sys.argv[1:7]
    extra = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else None
    in_cand = sys.argv[8] if len(sys.argv) > 8 else None
    os.makedirs(out_dir, exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id"])
    q = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = q.select("rid", pl.col("entity_id").alias("m"))
    r = pl.read_parquet(resc).sort("p_rescue", descending=True).group_by("rid").first()
    r = r.with_columns((pl.col("p_rescue") / ODDS_K / (pl.col("p_rescue") / ODDS_K + 1 - pl.col("p_rescue"))).alias("p2")).filter(pl.col("p2") >= MIN_P)
    if extra:
        ex = pl.read_parquet(extra).select("rid", "s1").join(r.select("rid"), on="rid", how="anti").with_columns(
            pl.lit(P_NOISE).alias("p_rescue"), pl.lit(P_NOISE).alias("p2"))
        r = pl.concat([r.select("rid", "s1", "p_rescue", "p2"), ex.select("rid", "s1", "p_rescue", "p2")])
        print("extra France pairs", ex.height, flush=True)
    aff = r.select("s1").unique()
    # stage-2 scores of the other candidates of the affected entities (v17: the cross-encoder-stacked scores, env BEST)
    best_path = os.environ.get("BEST", f"{W}/test_best_s2v2_cf_final.parquet")
    best = pl.read_parquet(best_path).select("rid", "s1", "p2").join(aff, on="s1")
    rem = (pl.read_csv(removed_tsv, separator="\t", infer_schema=False, quote_char=None)
             .select("source1_entity_id", pl.col("removed_entity_id").alias("m"))
             .join(s1id, on="source1_entity_id").join(qid, on="m").select("rid", "s1", pl.lit(True).alias("removed")))
    fx = pl.read_parquet(ff2).select("rid", "s1", "DECOY", "NOISE")
    best = (best.join(rem, on=["rid", "s1"], how="left").join(fx, on=["rid", "s1"], how="left")
                .with_columns(pl.col("removed").fill_null(False), pl.col("DECOY").fill_null(False), pl.col("NOISE").fill_null(False))
                .with_columns(pl.when(pl.col("removed") | pl.col("DECOY")).then(0.0)
                                .when(pl.col("NOISE")).then(pl.max_horizontal(pl.col("p2"), pl.lit(P_NOISE)))
                                .otherwise(pl.col("p2")).alias("p2")))
    pool = pl.concat([best.select("rid", "s1", "p2"), r.select("rid", "s1", pl.col("p2").cast(pl.Float64))])
    new_sel = ev_select(pool, "p2")
    sub = pl.read_csv(in_tsv, separator="\t", infer_schema=False, quote_char=None)
    old = (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
              .join(s1id, on="source1_entity_id").join(qid, on="m").select("rid", "s1"))
    keep = old.join(aff, on="s1", how="anti")
    acc = pl.concat([keep, new_sel.select("rid", "s1")]).unique()
    n_resc = new_sel.join(r.select("rid", "s1"), on=["rid", "s1"]).height
    print("rescued pairs offered", r.height, "affected S1", aff.height, "| selected rescued pairs", n_resc,
          "| pairs in", old.height, "out", acc.height, flush=True)
    m = (s1id.join(acc.join(qid, on="rid").group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
             .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    m = sub.select("source1_entity_id").join(m, on="source1_entity_id", how="left", maintain_order="left")
    m.write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    rs = new_sel.join(r.select("rid", "s1", "p_rescue"), on=["rid", "s1"])
    rs.write_parquet(f"{out_dir}/rescued_pairs.parquet")
    if in_cand:
        c = pl.read_csv(in_cand, separator="\t", infer_schema=False, quote_char=None)
        have = (c.filter(pl.col("candidate_entity_ids").is_not_null() & (pl.col("candidate_entity_ids") != ""))
                  .with_columns(pl.col("candidate_entity_ids").str.split(",").alias("m")).explode("m").select("source1_entity_id", "m"))
        # every rescued pair that entered the final selection (not only the selected ones)
        add = (r.select("rid", "s1").join(s1id, on="s1").join(qid, on="rid").select("source1_entity_id", "m")
                 .join(have, on=["source1_entity_id", "m"], how="anti")
                 .group_by("source1_entity_id").agg(pl.col("m").sort().str.join(",").alias("add")))
        c2 = c.join(add, on="source1_entity_id", how="left", maintain_order="left").with_columns(
            pl.when(pl.col("add").is_null()).then(pl.col("candidate_entity_ids"))
              .when(pl.col("candidate_entity_ids").is_null() | (pl.col("candidate_entity_ids") == "")).then(pl.col("add"))
              .otherwise(pl.concat_str([pl.col("candidate_entity_ids"), pl.col("add")], separator=",")).alias("candidate_entity_ids")).drop("add")
        c2.write_csv(f"{out_dir}/candidate_pairs.tsv", separator="\t", quote_style="never")
        print("candidate pairs added", add.select(pl.col("add").str.split(",").list.len().sum()).item(), flush=True)
    print("written", f"{out_dir}/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main()
