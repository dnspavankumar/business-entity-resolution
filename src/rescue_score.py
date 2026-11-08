"""Features and model for the rescue candidates (records that the main blocking gave no candidate).

features(split, cand) -> one row per (rid, s1) candidate with name / address similarity features.
Model: LightGBM, trained on train no-candidate records (label = the candidate is the true entity),
2-fold cross-fitted by record hash so that every train record gets an out-of-fold score.

usage: python rescue_score.py <work_dir> {train|predict}
  train   : builds train features, cross-fits, writes rescue_oof_train.parquet and model_rescue.txt
  predict : builds test features and writes rescue_pred_test.parquet (rid, s1, p_rescue)
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

t0 = time.time()
W = sys.argv[1]
FEATS = ["name_sim", "addr_sim", "sum_sim", "r_sum", "r_name", "r_addr", "gap_sum", "gap_name", "n_cand",
         "fz_ratio", "fz_tset", "fz_partial", "fz_core", "lev_core", "num_jac", "num_first_eq", "num_any",
         "city_ov", "tok_ov", "q_len", "s_len", "len_ratio", "is_domain", "is_native", "q_ntok", "s_ntok",
         "core_in_q", "q_in_core"]


def log(*a):
    print(*a, f"{time.time() - t0:.0f}s", flush=True)


def features(split: str, cand: pl.DataFrame) -> pl.DataFrame:
    cols = ["rid", "nfull", "ncore", "anums", "atoks", "acity"]
    q = pl.read_parquet(f"{W}/{split}_q.parquet", columns=cols + ["is_domain", "is_native"]).join(cand.select("rid").unique(), on="rid")
    s = (pl.read_parquet(f"{W}/{split}_s1.parquet", columns=cols).join(cand.select(pl.col("s1").alias("rid")).unique(), on="rid")
           .rename({c: "s_" + c for c in cols if c != "rid"}).rename({"rid": "s1"}))
    x = cand.join(q, on="rid", how="left").join(s, on="s1", how="left")
    x = x.with_columns((pl.col("name_sim") + pl.col("addr_sim")).alias("sum_sim"))
    x = x.with_columns(pl.col("sum_sim").rank("ordinal", descending=True).over("rid").alias("r_sum"),
                       pl.col("name_sim").rank("ordinal", descending=True).over("rid").alias("r_name"),
                       pl.col("addr_sim").rank("ordinal", descending=True).over("rid").alias("r_addr"),
                       pl.len().over("rid").alias("n_cand"))
    # gap between this candidate and the best OTHER candidate of the same record
    for c in ("sum_sim", "name_sim"):
        top2 = x.group_by("rid").agg(pl.col(c).top_k(2).alias("_t"))
        top2 = top2.with_columns(pl.col("_t").list.get(0, null_on_oob=True).alias("_t1"), pl.col("_t").list.get(1, null_on_oob=True).fill_null(0.0).alias("_t2")).drop("_t")
        x = x.join(top2, on="rid").with_columns(
            pl.when(pl.col(c) >= pl.col("_t1")).then(pl.col(c) - pl.col("_t2")).otherwise(pl.col(c) - pl.col("_t1")).alias("gap_" + c.split("_")[0])).drop("_t1", "_t2")
    qn, sn = x["nfull"].fill_null("").to_list(), x["s_nfull"].fill_null("").to_list()
    qc, sc = x["ncore"].fill_null("").to_list(), x["s_ncore"].fill_null("").to_list()
    x = x.with_columns(
        pl.Series("fz_ratio", [fuzz.ratio(a, b) for a, b in zip(qn, sn)], dtype=pl.Float32),
        pl.Series("fz_tset", [fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)], dtype=pl.Float32),
        pl.Series("fz_partial", [fuzz.partial_ratio(a.replace(" ", ""), b.replace(" ", "")) for a, b in zip(qn, sn)], dtype=pl.Float32),
        pl.Series("fz_core", [fuzz.ratio(a, b) for a, b in zip(qc, sc)], dtype=pl.Float32),
        pl.Series("lev_core", [Levenshtein.normalized_similarity(a.replace(" ", ""), b.replace(" ", "")) for a, b in zip(qc, sc)], dtype=pl.Float32),
        pl.Series("core_in_q", [float(bool(b) and b.replace(" ", "") in a.replace(" ", "")) for a, b in zip(qn, sc)], dtype=pl.Float32),
        pl.Series("q_in_core", [float(bool(a) and a.replace(" ", "") in b.replace(" ", "")) for a, b in zip(qc, sn)], dtype=pl.Float32))
    x = x.with_columns(pl.col("anums").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("qn_"),
                       pl.col("s_anums").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("sn_"),
                       pl.col("acity").fill_null("").str.split("|").alias("qc_"), pl.col("s_acity").fill_null("").str.split("|").alias("sc_"),
                       pl.col("atoks").fill_null("").str.split(" ").alias("qt_"), pl.col("s_atoks").fill_null("").str.split(" ").alias("st_"))
    x = x.with_columns(
        (pl.col("qn_").list.set_intersection(pl.col("sn_")).list.len() / pl.col("qn_").list.set_union(pl.col("sn_")).list.len().clip(1, None)).cast(pl.Float32).alias("num_jac"),
        ((pl.col("qn_").list.first() == pl.col("sn_").list.first()) & (pl.col("qn_").list.len() > 0)).fill_null(False).cast(pl.Float32).alias("num_first_eq"),
        (pl.col("qn_").list.set_intersection(pl.col("sn_")).list.len() > 0).cast(pl.Float32).alias("num_any"),
        (pl.col("qc_").list.set_intersection(pl.col("sc_")).list.eval(pl.element().filter(pl.element() != "")).list.len() > 0).cast(pl.Float32).alias("city_ov"),
        (pl.col("qt_").list.set_intersection(pl.col("st_")).list.len() / pl.col("qt_").list.len().clip(1, None)).cast(pl.Float32).alias("tok_ov"),
        pl.col("nfull").fill_null("").str.len_chars().cast(pl.Float32).alias("q_len"),
        pl.col("s_nfull").fill_null("").str.len_chars().cast(pl.Float32).alias("s_len"),
        pl.col("nfull").fill_null("").str.split(" ").list.len().cast(pl.Float32).alias("q_ntok"),
        pl.col("s_nfull").fill_null("").str.split(" ").list.len().cast(pl.Float32).alias("s_ntok"))
    x = x.with_columns((pl.min_horizontal("q_len", "s_len") / pl.max_horizontal("q_len", "s_len").clip(1, None)).alias("len_ratio"),
                       pl.col("is_domain").fill_null(0).cast(pl.Float32), pl.col("is_native").fill_null(0).cast(pl.Float32))
    return x.select("rid", "s1", *FEATS)


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=7)
ROUNDS = 400


def main():
    step = sys.argv[2]
    if step == "train":
        cand = pl.read_parquet(f"{W}/rescue_cand_train.parquet")
        x = features("train", cand)
        gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").rename({"true_s1": "s1"}).with_columns(pl.lit(1).alias("label"))
        x = x.join(gt, on=["rid", "s1"], how="left").with_columns(pl.col("label").fill_null(0))
        x = x.with_columns((pl.col("rid").hash(7) % 2).alias("fold"))
        log("train rows", x.height, "positives", int(x["label"].sum()))
        oof = []
        for f in (0, 1):
            tr, te = x.filter(pl.col("fold") != f), x.filter(pl.col("fold") == f)
            m = lgb.train(PARAMS, lgb.Dataset(tr.select(FEATS).to_numpy(), label=tr["label"].to_numpy()), num_boost_round=ROUNDS)
            oof.append(te.select("rid", "s1", "label").with_columns(pl.Series("p_rescue", m.predict(te.select(FEATS).to_numpy()))))
        oof = pl.concat(oof)
        oof.write_parquet(f"{W}/rescue_oof_train.parquet")
        m = lgb.train(PARAMS, lgb.Dataset(x.select(FEATS).to_numpy(), label=x["label"].to_numpy()), num_boost_round=ROUNDS)
        m.save_model(f"{W}/model_rescue.txt")
        imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda t: -t[1])[:12]
        log("importance", [(a, int(b)) for a, b in imp])
    else:
        cand = pl.read_parquet(f"{W}/rescue_cand_test.parquet")
        x = features("test", cand)
        m = lgb.Booster(model_file=f"{W}/model_rescue.txt")
        x.select("rid", "s1").with_columns(pl.Series("p_rescue", m.predict(x.select(FEATS).to_numpy()))).write_parquet(f"{W}/rescue_pred_test.parquet")
        log("written test predictions", x.height)


if __name__ == "__main__":
    main()
