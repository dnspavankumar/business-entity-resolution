"""Export the rescue step's top-5 candidates per record (train: out-of-fold rescue scores with labels, test) as text pairs
for the GPU cross-encoder (gpu_crossencoder/xenc3_rescue_colab.ipynb): xextra_part00/01.parquet with rid, s1, a, b, split.
usage: python xenc_export_rescue.py <work_dir> <out_dir>"""
import os
import sys

import polars as pl

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
K = 5


def text(df, prefix):
    n = pl.col("business_name").fill_null("").str.replace_all(r"[\t\n\r]", " ")
    a = pl.col("business_address").fill_null("").str.replace_all(r"[\t\n\r]", " ")
    return df.select("rid", pl.concat_str([n, pl.lit(" | "), a]).alias(prefix))


def build(pairs, split):
    q = text(pl.read_parquet(f"{W}/{split}_q.parquet", columns=["rid", "business_name", "business_address"]).join(pairs.select("rid").unique(), on="rid"), "a")
    s = text(pl.read_parquet(f"{W}/{split}_s1.parquet", columns=["rid", "business_name", "business_address"]).join(pairs.select(pl.col("s1").alias("rid")).unique(), on="rid"), "b")
    return pairs.join(q, on="rid").join(s.rename({"rid": "s1"}), on="s1")


top = lambda d: d.with_columns(pl.col("p_rescue").rank("ordinal", descending=True).over("rid").alias("r")).filter(pl.col("r") <= K)
tr = top(pl.read_parquet(f"{W}/rescue_oof_train.parquet"))
te = top(pl.read_parquet(f"{W}/rescue_pred_test.parquet"))
x = pl.concat([build(tr.select("rid", "s1"), "train").with_columns(pl.lit("rtrain").alias("split")),
               build(te.select("rid", "s1"), "test").with_columns(pl.lit("rtest").alias("split"))]).sample(fraction=1.0, shuffle=True, seed=3)
half = (x.height + 1) // 2
for i in range(2):
    x.slice(i * half, half).write_parquet(f"{OUT}/xextra_part{i:02d}.parquet", compression="zstd", compression_level=19)
print("pairs", x.height)
