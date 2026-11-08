"""Convert the challenge TSVs to parquet.  usage: python convert.py <dataset_dir> <data_out_dir>"""
import sys, os
import polars as pl

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for split in ("train", "test"):
    for name in [f"{split}_source1", f"{split}_source2", f"{split}_source3"] + (["train_ground_truth"] if split == "train" else []):
        df = pl.read_csv(f"{src}/{split}/{name}.tsv", separator="\t", quote_char=None, infer_schema=False)
        df = df.with_columns(pl.all().fill_null(""))
        df.write_parquet(f"{out}/{name}.parquet")
        print(name, df.shape, flush=True)
