"""Export (record, Source-1) text pairs for the GPU cross-encoder (colab/xenc_colab.ipynb).
  xtrain : T2+T3+T4 best pairs with 0.02 <= p1 <= 0.999 (+ 60k random easy ones), label
  xvt    : V+T best pairs in the uncertain band of the cross-fitted v15 score (local check / stacking), label
  xtest  : test best pairs in the uncertain band of the v15 score
usage: python xenc_export.py <work_dir> <out_dir>"""
import os, sys
import polars as pl

W, OUT = sys.argv[1], sys.argv[2]
V15 = os.environ.get("V15_DIR", f"{W}/v15")   # folder with the matcher's cross-fit preds + test scores (v16 in run.sh)
os.makedirs(OUT, exist_ok=True)


def text(df, prefix):
    n = pl.col("business_name").fill_null("").str.replace_all(r"[\t\n\r]", " ")
    a = pl.col("business_address").fill_null("").str.replace_all(r"[\t\n\r]", " ")
    return df.select("rid", pl.concat_str([n, pl.lit(" | "), a]).alias(prefix))


def build(pairs, split):
    q = text(pl.read_parquet(f"{W}/{split}_q.parquet", columns=["rid", "business_name", "business_address"]).join(pairs.select("rid").unique(), on="rid"), "a")
    s = text(pl.read_parquet(f"{W}/{split}_s1.parquet", columns=["rid", "business_name", "business_address"]).join(pairs.select(pl.col("s1").alias("rid")).unique(), on="rid"), "b")
    return pairs.join(q, on="rid").join(s.rename({"rid": "s1"}), on="s1")


def write(df, name, rows_per_part=180_000):
    for i in range(0, df.height, rows_per_part):
        p = f"{OUT}/{name}_part{i // rows_per_part:02d}.parquet"
        df.slice(i, rows_per_part).write_parquet(p, compression="zstd", compression_level=19)
        print(p, os.path.getsize(p) // 1_000_000, "MB", flush=True)


tr = pl.scan_parquet(f"{W}/s2v2_train_v14.parquet").filter(pl.col("isT2")).select("rid", "s1", "p1", "label").collect()
hard = tr.filter((pl.col("p1") >= 0.02) & (pl.col("p1") <= 0.999))
easy = tr.filter((pl.col("p1") < 0.02) | (pl.col("p1") > 0.999)).sample(60_000, seed=1)
xtr = build(pl.concat([hard, easy]).select("rid", "s1", pl.col("label").cast(pl.Int8)), "train").sample(fraction=1.0, shuffle=True, seed=2)
write(xtr, "xtrain")
vt = pl.read_parquet(f"{V15}/cf_preds_v14_T2.parquet").filter((pl.col("p2") > 0.002) & (pl.col("p2") < 0.998)).select("rid", "s1", pl.col("label").cast(pl.Int8))
write(build(vt, "train"), "xvt")
te = pl.read_parquet(f"{V15}/test_best_s2v2_cf_final.parquet")
te = te.filter(((pl.col("p2") > 0.002) & (pl.col("p2") < 0.998)) | ((pl.col("p2_raw") > 0.002) & (pl.col("p2_raw") < 0.998))).select("rid", "s1")
write(build(te, "test"), "xtest")
print("train", xtr.height, "vt", vt.height, "test", te.height)
