"""Compute pairwise features for candidate pairs.

usage: python build_features.py <split> <work_dir> [mode]
  split=train     : V (3% of Source-1 entities, held out) + 2M random training queries
  split=trainT    : queries touching the T subset (see select_T.py)
  split=trainT2   : queries touching the T2 subset (see select_T2.py; extra stage-2 training data, v14)
  split=trainRest : every remaining train query (needed for full cross-fitting)
  split=test      : all test candidate pairs, per country
Features are written in chunks to work/feat/*_partNNNN.parquet
"""
import sys, time, gc, os, glob
import numpy as np
import polars as pl
from features import compute_features, token_idf, REC_COLS

split, W = sys.argv[1], sys.argv[2]
N_TRAIN_Q = int(os.environ.get("N_TRAIN_Q", "2000000"))
CHUNK = 600000

t0 = time.time()
os.makedirs(f"{W}/feat", exist_ok=True)
base = "train" if split.startswith("train") else split
idf_path_n, idf_path_a = f"{W}/{base}_idf_name.parquet", f"{W}/{base}_idf_addr.parquet"
if not os.path.exists(idf_path_a):
    frames = [pl.scan_parquet(f"{W}/{base}_s1.parquet"), pl.scan_parquet(f"{W}/{base}_q.parquet")]
    token_idf(frames, "ncore").write_parquet(idf_path_n)
    token_idf(frames, "atoks").write_parquet(idf_path_a)
name_idf = pl.read_parquet(idf_path_n).select("country", "tok", "idf")
addr_idf = pl.read_parquet(idf_path_a).select("country", "tok", "idf")
if split.startswith("train"):
    s1 = pl.read_parquet(f"{W}/train_s1.parquet", columns=REC_COLS + ["entity_id"])
    cand = pl.read_parquet(f"{W}/train_cand.parquet", columns=["rid", "s1", "bscore", "nkeys", "brank", "btop"])
else:
    s1 = cand = pl.DataFrame()
print("loaded", f"{time.time()-t0:.0f}s", flush=True)
print("idf ready", f"{time.time()-t0:.0f}s", flush=True)


def run(pairs, out_path, label_map=None):
    pairs = pairs.sort("rid")
    rids = pairs["rid"].unique().sort()
    parts = []
    # chunk by query id so all candidates of a query are processed together
    counts = pairs.group_by("rid").len().sort("rid")
    cum = counts["len"].cum_sum().to_numpy()
    bounds = [0]
    nxt = CHUNK
    for i, c in enumerate(cum):
        if c >= nxt:
            bounds.append(i + 1); nxt = c + CHUNK
    if bounds[-1] != len(cum):
        bounds.append(len(cum))
    rr = counts["rid"].to_numpy()
    for bi in range(len(bounds) - 1):
        lo, hi = rr[bounds[bi]], rr[bounds[bi + 1] - 1]
        pc = pairs.filter((pl.col("rid") >= lo) & (pl.col("rid") <= hi))
        f = compute_features(pc, q.filter(pl.col("rid").is_in(pc["rid"].unique())), s1.filter(pl.col("rid").is_in(pc["s1"].unique())), name_idf, addr_idf)
        if label_map is not None:
            f = f.join(label_map, on="rid", how="left").with_columns((pl.col("true_s1") == pl.col("s1")).fill_null(False).cast(pl.Int8).alias("label"))
        f = f.with_columns([pl.col(c).cast(pl.Float32) for c, dt in f.schema.items() if dt == pl.Float64])
        f.write_parquet(f"{out_path[:-8]}_part{bi:04d}.parquet")
        del f
        print(f"  chunk {bi+1}/{len(bounds)-1} pairs={pc.height} {time.time()-t0:.0f}s", flush=True)
        gc.collect()
    return None


if split == "trainRest":
    done = pl.concat([pl.scan_parquet(p).select("rid") for p in sorted(glob.glob(f"{W}/feat/train_*_part*.parquet"))]).unique().collect()
    qr = cand.select("rid").unique().join(done, on="rid", how="anti")
    print("rest queries", qr.height, flush=True)
    cr = cand.join(qr, on="rid").filter(pl.col("bscore") >= 0.3 * pl.col("btop"))
    del cand; gc.collect()
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet")
    rids = qr["rid"].sort()
    B = 1_200_000
    for bi in range(0, len(rids), B):
        rb = pl.DataFrame({"rid": rids[bi:bi + B]})
        q = pl.scan_parquet(f"{W}/train_q.parquet").select(REC_COLS).join(rb.lazy(), on="rid").collect()
        run(cr.join(rb, on="rid"), f"{W}/feat/train_rest{bi // B}.parquet", gt)
        del q; gc.collect()
elif split == "trainT2":   # extra stage-2 training subsets: SUB=T2 (default) or T3 ... (select_T2.py)
    SUB = os.environ.get("SUB", "T2")
    qt = pl.read_parquet(f"{W}/q{SUB}.parquet")
    q = pl.scan_parquet(f"{W}/train_q.parquet").select(REC_COLS).join(qt.lazy(), on="rid").collect()
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet")
    run(cand.join(qt, on="rid"), f"{W}/feat/train_{SUB}.parquet", gt)
elif split == "trainT":
    qt = pl.read_parquet(f"{W}/qT.parquet")
    q = pl.scan_parquet(f"{W}/train_q.parquet").select(REC_COLS).join(qt.lazy(), on="rid").collect()
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet")
    run(cand.join(qt, on="rid"), f"{W}/feat/train_T.parquet", gt)
elif split == "train":
    gt = pl.read_parquet(os.environ.get("GT", f"{W}/data/train_ground_truth.parquet"))
    gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
          .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("rid").alias("true_s1")), on="source1_entity_id")
          .join(pl.read_parquet(f"{W}/{split}_q.parquet", columns=["entity_id", "rid"]).rename({"entity_id": "m"}), on="m").select("rid", "true_s1"))
    # hold-out: 5% of Source-1 entities
    s1v = s1.filter((pl.col("entity_id").hash(seed=42) % 100) < 3).select(pl.col("rid").alias("s1"))
    s1v.write_parquet(f"{W}/val_s1.parquet")
    qv = cand.filter(pl.col("bscore") >= 0.3 * pl.col("btop")).join(s1v, on="s1").select("rid").unique()
    # also include queries whose TRUE s1 is in val but have no val candidate (for recall accounting only; no pairs)
    print("val s1", s1v.height, "val queries", qv.height, flush=True)
    rest = cand.select("rid").unique().join(qv, on="rid", how="anti")
    qt = rest.sample(min(N_TRAIN_Q, rest.height), seed=0)
    keep = pl.concat([qv, qt]).unique()
    q = pl.scan_parquet(f"{W}/{split}_q.parquet").select(REC_COLS).join(keep.lazy(), on="rid").collect()
    run(cand.join(qv, on="rid"), f"{W}/feat/train_val.parquet", gt)
    run(cand.join(qt, on="rid"), f"{W}/feat/train_trn.parquet", gt)
    gt.write_parquet(f"{W}/train_gt_pairs.parquet")
else:
    del cand, s1
    gc.collect()
    for country in pl.scan_parquet(f"{W}/{split}_s1.parquet").select(pl.col("country").unique()).collect()["country"].to_list():
        cand = (pl.scan_parquet(f"{W}/{split}_cand_{country}.parquet").select(["rid", "s1", "bscore", "nkeys", "brank", "btop"])
                .filter(pl.col("bscore") >= 0.3 * pl.col("btop")).collect())
        s1 = pl.scan_parquet(f"{W}/{split}_s1.parquet").filter(pl.col("country") == country).select(REC_COLS).collect()
        q = pl.scan_parquet(f"{W}/{split}_q.parquet").filter(pl.col("country") == country).select(REC_COLS).collect()
        print(country, "pairs", cand.height, flush=True)
        run(cand, f"{W}/feat/test_{country}.parquet")
        del cand, s1, q
        gc.collect()
print("done", f"{time.time()-t0:.0f}s")
