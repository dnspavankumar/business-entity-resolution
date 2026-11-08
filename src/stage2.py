"""Stage 2: acceptance model for each query's best Source-1 candidate, using cluster context.

For each query q (a Source-2/3 record) stage 1 gives p1 for every candidate.  The best
candidate x = argmax p1 is kept and stage 2 decides whether to accept (q -> x) using
  * the stage-1 features of (q, x), p1 and the margin to the runner-up
  * the context of x: the other queries whose best candidate is also x (how many, how
    confident, from which source)
  * query-query similarity between q and the confident members of x's cluster
    (records of one real entity resemble each other, even when one of them is far from
    the Source-1 record).
"""
import numpy as np
import polars as pl
from rapidfuzz import process, fuzz

CTX_MIN = 0.3    # a query joins x's context cluster if its best p1 >= CTX_MIN
MEMBER_MIN = 0.5  # members used for query-query similarity


def best_pairs(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: rid, s1, p1 -> one row per rid with best s1, p1, runner-up p1, #candidates>0.1."""
    srt = pred.sort(["rid", "p1"], descending=[False, True])
    g = srt.group_by("rid", maintain_order=True).agg(
        pl.col("s1").first().alias("s1"), pl.col("p1").first().alias("p1"),
        pl.col("p1").get(1, null_on_oob=True).fill_null(0.0).alias("p1_2nd"),
        (pl.col("p1") > 0.1).sum().alias("n_p1_gt01"), pl.len().alias("n_cand_all"))
    return g.with_columns((pl.col("p1") - pl.col("p1_2nd")).alias("p1_margin"))


def context_features(best: pl.DataFrame, src: pl.DataFrame) -> pl.DataFrame:
    """best: rid, s1, p1 ; src: rid, src (2/3).  Adds cluster context of s1 excluding the query itself."""
    b = best.join(src, on="rid", how="left")
    mem = b.filter(pl.col("p1") >= CTX_MIN)
    agg = mem.group_by("s1").agg(
        pl.len().alias("c_n"), pl.col("p1").sum().alias("c_sum"),
        (pl.col("p1") >= 0.9).sum().alias("c_n_hi"),
        (pl.col("src") == 2).sum().alias("c_n_s2"), (pl.col("src") == 3).sum().alias("c_n_s3"),
        pl.col("p1").sort(descending=True).head(2).alias("c_top2"))
    b = b.join(agg, on="s1", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in ("c_n", "c_sum", "c_n_hi", "c_n_s2", "c_n_s3")])
    self_in = (pl.col("p1") >= CTX_MIN)
    b = b.with_columns(
        (pl.col("c_n") - self_in.cast(pl.Int64)).alias("c_n_oth"),
        (pl.col("c_sum") - pl.when(self_in).then(pl.col("p1")).otherwise(0.0)).alias("c_sum_oth"),
        (pl.col("c_n_hi") - (pl.col("p1") >= 0.9).cast(pl.Int64)).alias("c_n_hi_oth"),
        (pl.when(pl.col("src") == 2).then(pl.col("c_n_s2")).otherwise(pl.col("c_n_s3")) - self_in.cast(pl.Int64)).alias("c_n_same_src"),
        pl.when(pl.col("src") == 2).then(pl.col("c_n_s3")).otherwise(pl.col("c_n_s2")).alias("c_n_other_src"),
        # best p1 among the other members
        pl.when(self_in & (pl.col("c_top2").list.first() == pl.col("p1")))
          .then(pl.col("c_top2").list.get(1, null_on_oob=True)).otherwise(pl.col("c_top2").list.first())
          .fill_null(0.0).alias("c_max_oth"),
    ).drop("c_top2", "c_n", "c_sum", "c_n_hi", "c_n_s2", "c_n_s3")
    return b


def qq_features(best: pl.DataFrame, qtext: pl.DataFrame) -> pl.DataFrame:
    """Similarity of each query to the other confident members of its best cluster.
    qtext: rid, ncore, atoks, anums."""
    mem = best.filter(pl.col("p1") >= MEMBER_MIN).select("s1", pl.col("rid").alias("rid2"))
    pr = best.select("rid", "s1").join(mem, on="s1").filter(pl.col("rid") != pl.col("rid2"))
    t1 = qtext.rename({"ncore": "n1", "atoks": "a1", "anums": "m1"})
    t2 = qtext.rename({"rid": "rid2", "ncore": "n2", "atoks": "a2", "anums": "m2"})
    if pr.height == 0:
        return best.select("rid").with_columns([pl.lit(-1.0).alias(c) for c in ("qq_n_tset", "qq_a_tset", "qq_sum", "qq_num_eq")])
    outs = []
    CH = 1_000_000
    for i in range(0, pr.height, CH):
        c = pr.slice(i, CH).join(t1, on="rid", how="left").join(t2, on="rid2", how="left")
        c = c.with_columns([pl.col(k).fill_null("") for k in ("n1", "a1", "m1", "n2", "a2", "m2")])
        ns = process.cpdist(c["n1"].to_list(), c["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=2, dtype=np.float32)
        as_ = process.cpdist(c["a1"].to_list(), c["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=2, dtype=np.float32)
        outs.append(c.select("rid", pl.Series("ns", ns), pl.Series("as", as_),
                             ((pl.col("m1") != "") & (pl.col("m1").str.split(" ").list.first() == pl.col("m2").str.split(" ").list.first())).cast(pl.Float32).alias("meq")))
    pr = pl.concat(outs)
    agg = pr.group_by("rid").agg(pl.col("ns").max().alias("qq_n_tset"), pl.col("as").max().alias("qq_a_tset"),
                                 (pl.col("ns") + pl.col("as")).max().alias("qq_sum"), pl.col("meq").max().alias("qq_num_eq"))
    return best.select("rid").join(agg, on="rid", how="left").with_columns(
        [pl.col(c).fill_null(-1.0) for c in ("qq_n_tset", "qq_a_tset", "qq_sum", "qq_num_eq")])


def candidate_competition(best: pl.DataFrame, cf: pl.DataFrame) -> pl.DataFrame:
    """cf: rid, s1, a_tset, nf_tset for ALL candidates of the queries.  How many other Source-1
    candidates look as good as the chosen one on name / on address (ambiguity signals)."""
    x = cf.join(best.select("rid", pl.col("s1").alias("bs1")), on="rid")
    oth = x.filter(pl.col("s1") != pl.col("bs1"))
    agg = oth.group_by("rid").agg(
        pl.col("a_tset").max().alias("cc_max_a_oth"), pl.col("nf_tset").max().alias("cc_max_n_oth"),
        (pl.col("a_tset") >= 90).sum().alias("cc_n_addr_hi_oth"), (pl.col("nf_tset") >= 90).sum().alias("cc_n_name_hi_oth"),
        ((pl.col("a_tset") >= 90) & (pl.col("nf_tset") >= 70)).sum().alias("cc_n_both_hi_oth"))
    return best.select("rid").join(agg, on="rid", how="left").with_columns(
        [pl.col(c).fill_null(-1 if c.startswith("cc_max") else 0) for c in
         ("cc_max_a_oth", "cc_max_n_oth", "cc_n_addr_hi_oth", "cc_n_name_hi_oth", "cc_n_both_hi_oth")])


def s1_duplicates(s1: pl.DataFrame) -> pl.DataFrame:
    """s1: rid, country, ncore, atoks -> how many Source-1 entities share the exact normalised name / address."""
    d = s1.with_columns(pl.len().over(["country", "ncore"]).alias("s1_name_dups"),
                        pl.when(pl.col("atoks") != "").then(pl.len().over(["country", "atoks"])).otherwise(0).alias("s1_addr_dups"))
    return d.select(pl.col("rid").alias("s1"), "s1_name_dups", "s1_addr_dups")


def build(pred: pl.DataFrame, qtext: pl.DataFrame, cf: pl.DataFrame = None, s1dups: pl.DataFrame = None) -> pl.DataFrame:
    """pred: rid, s1, p1 for all candidate pairs of a (context-complete) query set.
    qtext: rid, entity_id, ncore, atoks, anums.  Returns best-pair table with stage-2 context features."""
    best = best_pairs(pred)
    src = qtext.select("rid", pl.when(pl.col("entity_id").str.starts_with("S2")).then(2).otherwise(3).cast(pl.Int8).alias("src"))
    b = context_features(best, src)
    qq = qq_features(best, qtext.select("rid", "ncore", "atoks", "anums"))
    b = b.join(qq, on="rid", how="left")
    if cf is not None:
        b = b.join(candidate_competition(best, cf), on="rid", how="left")
    if s1dups is not None:
        b = b.join(s1dups, on="s1", how="left")
    return b
