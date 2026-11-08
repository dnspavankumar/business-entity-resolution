"""Final merge (v15): take the rows of some countries from one matching_results.tsv and the rest from another.
v15 uses the v15 matcher for US and India (checked on the local US/India harness) and keeps the v13 decisions
for France (no labels; the count-profile check showed the new matcher accepts more empty-address records of
same-name French twins, which are coin flips).
usage: python merge_countries.py <work_dir> <base.tsv (France etc.)> <new.tsv (US, India)> <out.tsv> [countries from new, default US,India]
"""
import sys

import polars as pl

W, base, new, out = sys.argv[1:5]
use_new = (sys.argv[5] if len(sys.argv) > 5 else "US,India").split(",")
s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
a = pl.read_csv(base, separator="\t", infer_schema=False, quote_char=None).fill_null("")
b = pl.read_csv(new, separator="\t", infer_schema=False, quote_char=None).fill_null("")
assert a.height == b.height and (a["source1_entity_id"] == b["source1_entity_id"]).all()
c = a.join(s1, on="source1_entity_id", how="left", maintain_order="left")
take_new = c["country"].is_in(use_new)
m = pl.select(pl.when(take_new).then(b["matched_entity_ids"]).otherwise(a["matched_entity_ids"])).to_series()
a.select("source1_entity_id").with_columns(m.alias("matched_entity_ids")).write_csv(out, separator="\t", quote_style="never")
print("rows from new file:", int(take_new.sum()), "| changed rows:", int((a["matched_entity_ids"] != m).sum()), "->", out)
