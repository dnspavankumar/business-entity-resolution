"""What kind of true pairs does the pipeline never reach? (local, labelled V+T entities, records WITH an address)
Uses <work_dir>/reach_true.parquet (case a = no candidate at all, b = true Source-1 not among the candidates,
c = true Source-1 not ranked first). Result 2026-09-27 (3,468 pairs):
  pseudo-name at a unique S1 address 1,024 | name shared by 2+ S1 921 | web-domain name 549 | native script 515 |
  pseudo-name at a shared address 310 | other 149
usage: python miss_kinds.py <work_dir> [n_examples]"""
import sys

import polars as pl

W = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60)
r = pl.read_parquet(f"{W}/reach_true.parquet").filter(~pl.col("ea") & pl.col("case").str.slice(0, 1).is_in(["a", "b", "c"]))
q = pl.read_parquet(f"{W}/train_q.parquet", columns=["rid", "nfull", "is_native", "is_domain", "business_name", "business_address"])
s = pl.read_parquet(f"{W}/train_s1.parquet", columns=["rid", "country", "ncore", "nfull", "anums", "atoks", "business_name", "business_address"]).rename({"rid": "s1"})
twin = s.group_by(["country", "ncore"]).len().rename({"len": "n_core"})
s = s.with_columns(pl.concat_str([pl.col("anums").fill_null(""), pl.lit("|"), pl.col("atoks").fill_null("")]).alias("akey"))
ak = s.group_by(["country", "akey"]).len().rename({"len": "n_addr"})
x = (r.join(q, on="rid")
      .join(s.select("s1", pl.col("country").alias("c2"), pl.col("ncore").alias("s_ncore"), "akey", pl.col("nfull").alias("s_nfull"),
                     pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="s1")
      .join(twin, left_on=["c2", "s_ncore"], right_on=["country", "ncore"], how="left")
      .join(ak, left_on=["c2", "akey"], right_on=["country", "akey"], how="left"))
tok = lambda c: pl.col(c).fill_null("").str.split(" ")
x = x.with_columns((tok("nfull").list.set_intersection(tok("s_nfull")).list.len() > 0).alias("name_overlap"))
x = x.with_columns(pl.when(pl.col("is_native").cast(pl.Boolean)).then(pl.lit("native script"))
                     .when(pl.col("is_domain").cast(pl.Boolean)).then(pl.lit("web-domain name"))
                     .when(~pl.col("name_overlap") & (pl.col("n_addr") >= 2)).then(pl.lit("pseudo-name, shared address"))
                     .when(~pl.col("name_overlap")).then(pl.lit("pseudo-name, unique address"))
                     .when(pl.col("n_core") >= 2).then(pl.lit("name shared by 2+ S1"))
                     .otherwise(pl.lit("other (unique name)")).alias("kind"))
print("missed true pairs with an address:", x.height)
print(x.group_by("kind").len().sort("len", descending=True))
print(x.group_by(["case", "kind"]).len().sort(["case", "len"], descending=[False, True]))
if N:
    for k in x["kind"].unique().sort().to_list():
        print(k); print(x.filter(pl.col("kind") == k).sample(min(N, x.filter(pl.col("kind") == k).height), seed=1)
                        .select("case", "business_name", "business_address", "s1_name", "s1_addr"))
