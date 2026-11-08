"""France-specific correction on top of a finished submission (v10 -> v11 experiment).

Why: France has no training labels. Two kinds of France pairs look like clear matches under the data
generator's rules, but the US/India-trained model rejects most of them:
  F1  same address (same first house number, same street words, same city) and the record name differs
      from the S1 name by ONE word that is a "noise word": a word that is (almost) never used by the
      France decoys that carry the shifted-house-number fingerprint (+3,+4,+5,+7,+9,+11,+13,+21).
      Examples: "Pages Sportive SCI" -> "Pages Parents SCI", "X Lycee EI" -> "X EI Services",
      "Saint-Herblain Parents" -> "Saint-Herblain & Associes". In US/India train the same operation
      (one word replaced by a generic word such as Services/Center/Partners at the same address) is
      99.9% true. The model rejects the France cases because stage 1 reacts to the IDF weight of the
      replaced word (feature wn_qmiss), which is a France artefact.
  F2  exact core name, same house number, same city, but a different street name. In US/India train
      this is 97.6-99.9% true; the model gives France cases p2 ~0.34.
Pairs whose record ADDS a legal suffix are excluded (legal-suffix changes are a decoy operation).

The rule raises p2 for these pairs (F1 -> 0.97, F2 -> 0.95), keeps every pair that the decoy
post-filter removed out of the selection, re-runs the expected-F0.5 selection only for the Source-1
entities that contain such pairs, and keeps every other entity's selection unchanged.

usage: python france_fix.py <work_dir> <in_matching_results.tsv> <removed_pairs.tsv> <out_dir>
  work_dir must contain test_best_s2v2_cf_final.parquet, test_q.parquet, test_s1.parquet and
  test_best_feats_France.parquet (decoy features of the France best pairs).
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoy_features as D  # noqa: E402

STRONG = [3, 4, 5, 7, 9, 11, 13, 21]
P_F1, P_F2 = 0.97, 0.95
MIN_SAME, MAX_OFF_RATIO = 30, 0.08


def ev_select(best: pl.DataFrame, col: str = "p2", max_n: int = 12, fallback_thr: float = 0.8) -> pl.DataFrame:
    import numpy as np
    b = best.select("rid", "s1", col).sort(["s1", col], descending=[False, True])
    g = b.group_by("s1", maintain_order=True).agg(pl.col("rid"), pl.col(col).alias("p"), pl.len().alias("n"))
    tuples, frames = [], []
    for n in sorted(g["n"].unique().to_list()):
        gg = g.filter(pl.col("n") == n)
        if n > max_n:
            frames.append(gg.explode(["rid", "p"]).filter(pl.col("p") >= fallback_thr).select("rid", "s1"))
            continue
        z = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        T = z.sum(1)
        F = np.zeros((2 ** n, n + 1))
        F[:, 0] = (T == 0).astype(float)
        for k in range(1, n + 1):
            tp = z[:, :k].sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                P = tp / k
                R = np.where(T > 0, tp / T, 0.0)
                F[:, k] = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
        Pm = np.clip(np.array(gg["p"].to_list(), dtype=np.float64), 1e-6, 1 - 1e-6)
        prob = np.exp(np.log(Pm) @ z.T + np.log(1 - Pm) @ (1 - z).T)
        k = (prob @ F).argmax(1)
        rids, s1s = gg["rid"].to_list(), gg["s1"].to_list()
        for i in np.nonzero(k)[0]:
            for r in rids[i][: k[i]]:
                tuples.append((r, s1s[i]))
    out = [pl.DataFrame(tuples, schema={"rid": best.schema["rid"], "s1": best.schema["s1"]}, orient="row")] if tuples else []
    return pl.concat(out + frames) if (out or frames) else pl.DataFrame(schema={"rid": best.schema["rid"], "s1": best.schema["s1"]})


def main():
    W, in_tsv, removed_tsv, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    best = pl.read_parquet(f"{W}/test_best_s2v2_cf_final.parquet").select("rid", "s1", "p2")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id", "ncore", "nfull", "atoks", "anums", "acity"])
    q = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id", "ncore", "nfull", "atoks", "anums", "acity", "country"])
    fcols = ["rid", "s1", "d1", "street_jac", "city_overlap", "name_exact_core", "legal_added"]
    if os.path.exists(f"{W}/test_best_feats_France.parquet"):
        fr = pl.read_parquet(f"{W}/test_best_feats_France.parquet", columns=fcols)
    else:   # decoy features of the France best pairs (same code as stage2_v2 uses)
        qf = q.filter(pl.col("country") == "France")
        vocab = D.s1_vocab(pl.read_parquet(f"{W}/test_s1.parquet"))
        fr = D.compute(best.join(qf.select("rid"), on="rid").select("rid", "s1"), qf, s1, vocab).select(fcols)
    ops = D.op_tokens(fr.select("rid", "s1"), q.filter(pl.col("country") == "France"), s1)
    fr = fr.join(ops, on=["rid", "s1"]).with_columns(
        pl.col("added").list.len().alias("na"), pl.col("dropped").list.len().alias("nd"),
        pl.col("added").list.first().alias("w"))
    same = (pl.col("d1") == 0) & (pl.col("street_jac") >= 0.99) & (pl.col("city_overlap") == 1)
    single = (pl.col("na") == 1) & (pl.col("nd") <= 1)
    # word statistics over France: how often a one-word change uses w with the decoy offset vs at the same address
    ws = fr.filter(single).group_by("w").agg(pl.col("d1").is_in(STRONG).sum().alias("w_off"), same.sum().alias("w_same"))
    fr = fr.join(ws, on="w", how="left")
    f1 = same & single & (pl.col("w_same") >= MIN_SAME) & (pl.col("w_off") <= MAX_OFF_RATIO * pl.col("w_same")) & (pl.col("legal_added") != 1)
    f2 = (pl.col("name_exact_core") == 1) & (pl.col("d1") == 0) & (pl.col("street_jac") < 0.3) & (pl.col("city_overlap") == 1) & (pl.col("legal_added") != 1)
    fr = fr.with_columns(f1.fill_null(False).alias("F1"), f2.fill_null(False).alias("F2"))
    fix = fr.filter(pl.col("F1") | pl.col("F2")).select("rid", "s1", "F1", "F2")
    print("F1 pairs", fix["F1"].sum(), "F2 pairs", fix["F2"].sum(), flush=True)
    fix.write_parquet(f"{out_dir}/france_fix_pairs.parquet")

    # pairs removed by the decoy post-filter stay removed
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = q.select("rid", pl.col("entity_id").alias("m"))
    rem = (pl.read_csv(removed_tsv, separator="\t", infer_schema=False, quote_char=None)
             .select("source1_entity_id", pl.col("removed_entity_id").alias("m"))
             .join(s1id, on="source1_entity_id").join(qid, on="m").select("rid", "s1"))
    aff = fix.select("s1").unique()
    cand = (best.join(aff, on="s1").join(fix, on=["rid", "s1"], how="left")
                .join(rem.with_columns(pl.lit(True).alias("removed")), on=["rid", "s1"], how="left")
                .with_columns(pl.when(pl.col("removed").fill_null(False)).then(0.0)
                                .when(pl.col("F1").fill_null(False)).then(pl.max_horizontal(pl.col("p2"), pl.lit(P_F1)))
                                .when(pl.col("F2").fill_null(False)).then(pl.max_horizontal(pl.col("p2"), pl.lit(P_F2)))
                                .otherwise(pl.col("p2")).alias("p2n")))
    new_sel = ev_select(cand.select("rid", "s1", pl.col("p2n").alias("p2")), "p2")

    sub = pl.read_csv(in_tsv, separator="\t", infer_schema=False, quote_char=None)
    old = (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
              .join(s1id, on="source1_entity_id").join(qid, on="m").select("rid", "s1"))
    keep = old.join(aff, on="s1", how="anti")
    acc = pl.concat([keep, new_sel.select("rid", "s1")]).unique()
    print("pairs: old", old.height, "new", acc.height, "| affected S1", aff.height,
          "| old sel in affected", old.join(aff, on="s1").height, "new sel in affected", new_sel.height, flush=True)
    m = (s1id.join(acc.join(qid, on="rid").group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
             .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    # keep the row order of the input file
    m = sub.select("source1_entity_id").join(m, on="source1_entity_id", how="left", maintain_order="left")
    m.write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    print("written", f"{out_dir}/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main()
