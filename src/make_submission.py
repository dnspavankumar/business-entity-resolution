"""Score the test set with both stages and write the two submission files.

usage: python make_submission.py <work_dir> <out_dir>
"""
import sys, glob, json, os, time
import numpy as np
import polars as pl
import lightgbm as lgb
import stage2
from train import read_part

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
CF = os.path.exists(f"{W}/model_stage2_cf.txt")   # cross-fitted models (2 stage-1 fold models, averaged)
if CF:
    m1_meta = json.load(open(f"{W}/model_stage1_cf.json"))
    m1_files = [f"{W}/model_stage1_f0.txt", f"{W}/model_stage1_f1.txt"]
    m2_meta = json.load(open(f"{W}/model_stage2_cf.json")); m2_file = f"{W}/model_stage2_cf.txt"
else:
    m1_meta = json.load(open(f"{W}/model_stage1.json")); m1_files = [f"{W}/model_stage1.txt"]
    m2_meta = json.load(open(f"{W}/model_stage2.json")); m2_file = f"{W}/model_stage2.txt"
THR = float(os.environ.get("THR2", m2_meta["thr"]))
countries = pl.scan_parquet(f"{W}/test_s1.parquet").select(pl.col("country").unique()).collect()["country"].to_list()

# ---- stage 1 scores for every candidate pair
pred_path = f"{W}/pred_test{'_cf' if CF else ''}.parquet"
if not os.path.exists(pred_path):
    m1s = [lgb.Booster(model_file=f) for f in m1_files]
    res = []
    for c in countries:
        for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet")):
            d = read_part(p)
            X = d.select(m1_meta["cols"]).to_numpy().astype(np.float32)
            p1 = np.mean([m.predict(X, num_threads=2) for m in m1s], axis=0)
            res.append(d.select("rid", "s1").with_columns(pl.Series("p1", p1).cast(pl.Float32)))
        print("stage1", c, f"{time.time()-t0:.0f}s", flush=True)
    pl.concat(res).write_parquet(pred_path)
print("stage1 ready", f"{time.time()-t0:.0f}s", flush=True)

# ---- stage 2 on each query's best candidate (per country: clusters never cross countries)
best_path = f"{W}/test_best_pred{'_cf' if CF else ''}.parquet"
m2 = lgb.Booster(model_file=m2_file)
P1_MIN = 0.02  # stage 2 is only trained on best pairs with p1 >= P1_MIN; below it the pair is rejected
s1dups = stage2.s1_duplicates(pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "country", "ncore", "atoks"]))
outs = []
for c in countries:
    qtext = pl.scan_parquet(f"{W}/test_q.parquet").filter(pl.col("country") == c).select("rid", "entity_id", "ncore", "atoks", "anums").collect()
    pred_c = pl.scan_parquet(pred_path).join(qtext.select("rid").lazy(), on="rid").collect()
    cf = pl.concat([read_part(p).select("rid", "s1", "a_tset", "nf_tset") for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet"))])
    best = stage2.build(pred_c, qtext, cf, s1dups)
    del cf
    del pred_c, qtext
    for p in sorted(glob.glob(f"{W}/feat/test_{c}_part*.parquet")):
        f = best.join(read_part(p), on=["rid", "s1"], how="inner")
        outs.append(f.select("rid", "s1", "p1").with_columns(
            pl.Series("p2", m2.predict(f.select(m2_meta["cols"]).to_numpy().astype(np.float32), num_threads=2)).cast(pl.Float32)))
    del best
    print("stage2", c, f"{time.time()-t0:.0f}s", flush=True)
best = pl.concat(outs)
best = best.with_columns(pl.when(pl.col("p1") >= P1_MIN).then(pl.col("p2")).otherwise(0.0).alias("p2"))
best.write_parquet(best_path)
print("stage2 done", f"{time.time()-t0:.0f}s", flush=True)

# ---- write files
s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id"]).rename({"rid": "s1", "entity_id": "source1_entity_id"})
qid = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
acc = best.filter((pl.col("p2") >= THR) & (pl.col("p1") >= P1_MIN)).select("rid", "s1").join(qid, on="rid")
match = (s1.join(acc.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
         .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
# candidate_pairs.tsv = exactly the pairs the final matcher (stage 2) scores: each record's
# top-ranked Source-1 candidate from the learned candidate ranker (stage 1), kept if p1 >= P1_MIN
cand = best.filter(pl.col("p1") >= P1_MIN).select("rid", "s1").join(qid, on="rid")
cp = (s1.join(cand.group_by("s1").agg(pl.col("entity_id").sort().str.join(",").alias("candidate_entity_ids")), on="s1", how="left")
      .select("source1_entity_id", pl.col("candidate_entity_ids").fill_null("")))
cp.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
n_match = acc.height
print("threshold", THR, "matched pairs", n_match, "S1 with >=1 match", match.filter(pl.col("matched_entity_ids") != "").height,
      "of", match.height, f"{time.time()-t0:.0f}s")
