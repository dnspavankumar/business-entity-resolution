"""Test-side decoy check by house-number offset cells, against the LOCAL true-pair reference.
True number typos are symmetric (+k as often as -k) but the model moves true +k pairs toward the uncertain band, so the
right reference for each cell is the plus/minus ratio of accepted TRUE pairs on the labelled local data.
excess = test plus - test minus * (local true plus / local true minus) ~ decoys accepted in the cell.
Result 2026-09-27 (v19): India ~2,750 excess +1/+2 pairs, spread over cells with decoy shares 0.1-0.3
(removal pays only above ~0.3) -> nothing safe to remove; US: no excess.
usage: python offset_cells.py <work_dir> <matching_results.tsv> <local_preds.parquet> <test_best_xenc.parquet> [K scale 1.5]"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from local_eval import LocalEval, ev_select, K_V10  # noqa: E402

W, TSV, LP, TB = sys.argv[1:5]
KS = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
le = LocalEval(W)
pv = pl.read_parquet(LP)
FE = ["d1", "ctx_s1num_confirmed", "ctx_same_offset", "legal_swapped", "legal_added", "n_add", "n_sub_vocab"]
ft = pl.scan_parquet(f"{W}/s2v2_train_v14.parquet").select(["rid", "s1"] + FE).join(pv.select("rid", "s1").lazy(), on=["rid", "s1"]).collect()
xv = pl.read_parquet(TB, columns=["rid", "s1"]).head(0)
cc = le.s1.select("s1", "country")
band_local = pl.read_csv(f"{W}/xenc_scores.csv", schema_overrides={"rid": pl.UInt32, "s1": pl.UInt32}).filter(pl.col("split") == "vt").select("rid", "s1", pl.lit(True).alias("band"))
rows = []
for c in ("US", "India"):
    k = K_V10[c] * KS
    p = pv.join(cc.filter(pl.col("country") == c), on="s1").with_columns((pl.col("p2x") * k / (pl.col("p2x") * k + 1 - pl.col("p2x"))).alias("_p"))
    a = (ev_select(p, "_p").join(p.select("rid", "s1", "label"), on=["rid", "s1"]).join(ft, on=["rid", "s1"], how="left")
           .join(band_local, on=["rid", "s1"], how="left").with_columns(pl.col("band").fill_null(False), pl.lit(c).alias("country"), pl.lit("local").alias("set")))
    rows.append(a)
loc = pl.concat(rows, how="diagonal_relaxed")
s1id = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "country"])
qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
sub = pl.read_csv(TSV, separator="\t", infer_schema=False, quote_char=None)
acc = (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
          .join(s1id.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("rid").alias("s1"), "country"), on="source1_entity_id")
          .join(qid.rename({"entity_id": "m"}), on="m").select("rid", "s1", "country"))
tf = pl.concat([pl.read_parquet(f"{W}/test_best_feats_{c}.parquet", columns=["rid", "s1"] + FE) for c in ("US", "India")])
xb = pl.read_parquet(TB).select("rid", "s1", pl.col("xenc0" if "xenc0" in pl.read_parquet_schema(TB) else "xenc").is_not_null().alias("band"))
te = (acc.filter(pl.col("country").is_in(["US", "India"])).join(tf, on=["rid", "s1"], how="left").join(xb, on=["rid", "s1"], how="left")
         .with_columns(pl.lit("test").alias("set"), pl.lit(None, dtype=pl.Int8).alias("label")))
al = pl.concat([loc.select(te.columns), te], how="diagonal_relaxed").filter(pl.col("d1").abs().is_in([1.0, 2.0]))
al = al.with_columns((pl.col("d1") > 0).alias("pos"), (pl.col("ctx_s1num_confirmed").fill_null(0) > 0).alias("s1_number_confirmed"),
                     (pl.col("ctx_same_offset").fill_null(0) > 0).alias("offset_shared"))


def cell(keys):
    out = []
    for c in ("US", "India"):
        L = al.filter((pl.col("set") == "local") & (pl.col("country") == c) & (pl.col("label") == 1))
        T = al.filter((pl.col("set") == "test") & (pl.col("country") == c))
        gl = L.group_by(keys).agg(pl.col("pos").sum().alias("local_plus"), (~pl.col("pos")).sum().alias("local_minus"))
        gt = T.group_by(keys).agg(pl.col("pos").sum().alias("test_plus"), (~pl.col("pos")).sum().alias("test_minus"))
        g = gt.join(gl, on=keys, how="left").with_columns(pl.lit(c).alias("country"))
        g = g.with_columns((pl.col("test_plus") - pl.col("test_minus") * pl.col("local_plus") / pl.col("local_minus")).round(0).alias("excess"))
        out.append(g.with_columns((pl.col("excess") / pl.col("test_plus")).round(2).alias("decoy_share")).sort(keys))
    print(pl.concat(out).select(["country"] + keys + ["test_plus", "test_minus", "local_plus", "local_minus", "excess", "decoy_share"]))


cell(["band"])
cell(["band", "s1_number_confirmed", "offset_shared"])
