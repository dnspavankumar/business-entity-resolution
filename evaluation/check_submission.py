"""Memory-light checks of candidate_pairs.tsv (the official validator needs >6 GB for a file this size)."""
import sys
import polars as pl
out, test_dir = sys.argv[1], sys.argv[2]
sc = lambda p: pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.all().fill_null(""))
c = sc(f"{out}/candidate_pairs.tsv"); m = sc(f"{out}/matching_results.tsv")
assert c.collect_schema().names() == ["source1_entity_id", "candidate_entity_ids"]
s1 = pl.scan_csv(f"{test_dir}/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False).select(pl.col("entity_id").alias("source1_entity_id"))
ids = c.select("source1_entity_id").collect()
n_s1 = s1.select(pl.len()).collect().item()
assert ids.height == n_s1 and ids["source1_entity_id"].n_unique() == n_s1, "row count / duplicate rows"
assert ids.lazy().join(s1, on="source1_entity_id", how="anti").select(pl.len()).collect().item() == 0
del ids
ce = (c.select(pl.col("source1_entity_id").cast(pl.Categorical), pl.col("candidate_entity_ids").str.split(",").alias("cid"))
      .explode("cid", empty_as_null=True).filter(pl.col("cid").is_not_null() & (pl.col("cid") != "")))
stats = ce.select(pl.len().alias("n"), pl.col("cid").str.contains(r"^S[23]-").all().alias("prefix_ok")).collect(engine="streaming")
print(stats)
dup = ce.group_by(["source1_entity_id", "cid"]).len().filter(pl.col("len") > 1).select(pl.len()).collect(engine="streaming").item()
me = (m.select(pl.col("source1_entity_id").cast(pl.Categorical), pl.col("matched_entity_ids").str.split(",").alias("cid"))
      .explode("cid", empty_as_null=True).filter(pl.col("cid").is_not_null() & (pl.col("cid") != "")))
miss = me.join(ce, on=["source1_entity_id", "cid"], how="anti").select(pl.len()).collect(engine="streaming").item()
print("duplicate ids within lists:", dup, " matched-not-in-candidates:", miss)
print("PASS" if (dup == 0 and miss == 0 and stats["prefix_ok"][0]) else "FAIL")
