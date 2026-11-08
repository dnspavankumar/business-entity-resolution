import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame) -> float:
    """pred/truth: (s1, rid) pairs; s1_ids: (s1) evaluation universe.  Macro F0.5 per Source-1 entity."""
    p = pred.group_by("s1").agg(pl.col("rid").alias("p"))
    t = truth.group_by("s1").agg(pl.col("rid").alias("t"))
    d = s1_ids.join(p, on="s1", how="left").join(t, on="s1", how="left")
    d = d.with_columns(pl.col("p").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))).list.len().alias("np"),
                       pl.col("t").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))).list.len().alias("nt"))
    d = d.with_columns(pl.when((pl.col("np") > 0) & (pl.col("nt") > 0))
                       .then(pl.col("p").list.set_intersection(pl.col("t")).list.len()).otherwise(0).alias("tp"))
    prec = pl.col("tp") / pl.col("np")
    rec = pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
         .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0)).then(0.0)
         .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    return d.select(f.mean()).item()


def assign(scores: pl.DataFrame, thr: float, col: str = "p") -> pl.DataFrame:
    """Each query (rid) goes to its best-scoring Source-1 candidate if score >= thr."""
    best = scores.sort(col, descending=True).group_by("rid").first()
    return best.filter(pl.col(col) >= thr).select("s1", "rid")
