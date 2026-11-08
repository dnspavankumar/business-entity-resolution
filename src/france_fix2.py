"""France correction v2 (v12): reject same-address "category swap" decoys, accept France noise-suffix records.

Why (found without labels, from the per-entity count profile of each pair class):
  For every class of France pairs we count, per Source-1 entity, how many records of that class it has,
  as a function of n_ref = the number of OTHER accepted records of the entity (a proxy for its true size).
  * A true-noise class grows with the entity size (singletons have ~none): profile ~[0.28,1,0.82,0.72,...].
  * A decoy class is generated once per entity, independent of its size (singletons have as many):
    flat profile ~[1.09,1,1.01,0.97,...]. The profiles of the US/India train classes confirm this with labels.
  France results:
  - Real-word swaps (the record replaces a business-type word of the S1 name by another one, e.g.
    "Dunkerque Club EURL" -> "Dunkerque College EURL", "Bordeaux Sportive SAS" -> "Bordeaux Soins SAS") and
    added business-type words have a flat profile: ~99% decoys ("another business at the same address").
    The model accepted ~11k of them (club/amicale/ecole/comite/center/union/maison) and v11 (france_fix.py F1)
    added ~16k more. India train has the same decoy operation (e.g. "Ram Infrastructure" -> "Ram Consultancy").
  - France noise suffixes ("drop a word + append Services / France / Developpement / Group / Co / & Associes /
    & Fils") have the true profile; the model rejected most "Services" and ~40% of "France" ones.
  - F2 of france_fix.py (exact name, same number, other street) has a flat profile: decoys. Not applied here.
Rules (France only):
  DECOY: the record adds a business-type word (France S1 vocabulary word, not a noise suffix, not a typo of a
         dropped word) and drops a vocabulary word (any address), or adds one at the same address -> p2 = 0.
  NOISE: one word added (and at most one dropped), the added word is a noise suffix, no legal suffix added,
         not DECOY -> p2 = max(p2, 0.97). Words: services/service/associes/france at the same address, at a
         near-same address (same number, street partly equal; + developpement), without house number, and for
         records with an empty address (+ developpement, group). Each of these classes has the true profile.
Pairs removed by the decoy post-filter stay removed. The expected-F0.5 selection is re-run only for the
entities with a changed pair; all other rows are kept from the input file (v10).

usage: python france_fix2.py <work_dir> <in_matching_results.tsv> <removed_pairs.tsv> <out_dir>
"""
import os
import sys

import polars as pl
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from france_fix import ev_select  # noqa: E402

NOISE = {"services", "service", "developpement", "france", "group", "co", "associes", "fils", "freres", "and", "et",
         "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc"}
NOISE_ACCEPT = {"services", "service", "associes", "france"}
P_NOISE = 0.97
VOCAB_MIN = 100


def _real_added(add, drop, V):
    """business-type words added by the record that are not a typo / abbreviation of a dropped word"""
    out = []
    for w in add:
        if w not in V:
            continue
        if any(Levenshtein.normalized_similarity(w, x) >= 0.5 or w.startswith(x) or x.startswith(w) for x in drop):
            continue
        out.append(w)
    return out


def classify(W: str) -> pl.DataFrame:
    best = pl.read_parquet(f"{W}/test_best_s2v2_cf_final.parquet").select("rid", "s1", "p2")
    feats = pl.read_parquet(f"{W}/test_best_feats_France.parquet", columns=["rid", "s1", "d1", "street_jac", "city_overlap", "legal_added"])
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "country", "nfull", "ncore"])
    q = (pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "country", "nfull", "business_address"]).filter(pl.col("country") == "France")
           .with_columns((pl.col("business_address").fill_null("").str.strip_chars() == "").alias("q_ea")).drop("country", "business_address"))
    s1f = s1.filter(pl.col("country") == "France")
    voc = (s1f.select(pl.col("ncore").fill_null("").str.split(" ").alias("t")).explode("t")
              .group_by("t").len().filter(pl.col("len") >= VOCAB_MIN))
    V = {w for w in voc["t"].to_list() if len(w) >= 4} - NOISE   # business-type words; short tokens are initials
    x = (feats.join(best, on=["rid", "s1"], how="left")
              .join(q, on="rid", how="left")
              .join(s1f.select(pl.col("rid").alias("s1"), pl.col("nfull").alias("s_nfull")), on="s1", how="left")
              .with_columns(pl.col("nfull").fill_null("").str.split(" ").alias("R"), pl.col("s_nfull").fill_null("").str.split(" ").alias("S"))
              .with_columns(pl.col("R").list.set_difference(pl.col("S")).alias("add"), pl.col("S").list.set_difference(pl.col("R")).alias("drop")))
    adds, drops = x["add"].to_list(), x["drop"].to_list()
    ra = [_real_added(a or [], d or [], V) for a, d in zip(adds, drops)]
    rd = [[w for w in (d or []) if w in V] for d in drops]
    x = x.with_columns(pl.Series("n_real_add", [len(r) for r in ra], dtype=pl.Int32),
                       pl.Series("n_real_drop", [len(r) for r in rd], dtype=pl.Int32))
    same = (pl.col("d1") == 0) & (pl.col("street_jac") >= 0.99) & (pl.col("city_overlap") == 1)
    decoy = ((pl.col("n_real_add") > 0) & (pl.col("n_real_drop") > 0)) | ((pl.col("n_real_add") > 0) & same)
    add_w = pl.col("add").list.set_difference(pl.lit(["and", "et"]))   # "& Associes" -> "and associes"
    w = add_w.list.first()
    ea = pl.col("q_ea")
    nonum = ~ea & (pl.col("d1").is_null() | pl.col("d1").is_nan())
    near = ~ea & (pl.col("d1") == 0) & (pl.col("street_jac") >= 0.3) & (pl.col("street_jac") < 0.99) & (pl.col("city_overlap") == 1)
    NA = list(NOISE_ACCEPT)
    noise = ((add_w.list.len() == 1) & (pl.col("drop").list.len() <= 1) & (pl.col("legal_added") != 1)
             & ((same & ~ea & w.is_in(NA)) | (near & w.is_in(NA + ["developpement"])) | (nonum & w.is_in(NA))
                | (ea & w.is_in(NA + ["developpement", "group"]))))
    return x.with_columns(decoy.fill_null(False).alias("DECOY"), (noise.fill_null(False) & ~decoy.fill_null(False)).alias("NOISE")).select(
        "rid", "s1", "p2", "DECOY", "NOISE")


def main():
    W, in_tsv, removed_tsv, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    x = classify(W)
    print("France best pairs", x.height, "DECOY", int(x["DECOY"].sum()), "NOISE", int(x["NOISE"].sum()), flush=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["rid", "entity_id"])
    q = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "entity_id"])
    s1id = s1.select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
    qid = q.select("rid", pl.col("entity_id").alias("m"))
    rem = (pl.read_csv(removed_tsv, separator="\t", infer_schema=False, quote_char=None)
             .select("source1_entity_id", pl.col("removed_entity_id").alias("m"))
             .join(s1id, on="source1_entity_id").join(qid, on="m").select("rid", "s1").with_columns(pl.lit(True).alias("removed")))
    sub = pl.read_csv(in_tsv, separator="\t", infer_schema=False, quote_char=None)
    old = (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
              .join(s1id, on="source1_entity_id").join(qid, on="m").select("rid", "s1").with_columns(pl.lit(True).alias("sel")))
    x = x.join(old, on=["rid", "s1"], how="left").join(rem, on=["rid", "s1"], how="left").with_columns(
        pl.col("sel").fill_null(False), pl.col("removed").fill_null(False))
    # entities whose selection can change: a selected DECOY pair, or an unselected NOISE pair
    aff = x.filter((pl.col("DECOY") & pl.col("sel")) | (pl.col("NOISE") & ~pl.col("sel") & ~pl.col("removed"))).select("s1").unique()
    best_all = pl.read_parquet(f"{W}/test_best_s2v2_cf_final.parquet").select("rid", "s1", "p2")
    cand = (best_all.join(aff, on="s1").join(x.select("rid", "s1", "DECOY", "NOISE", "removed"), on=["rid", "s1"], how="left")
                .with_columns(pl.col("DECOY").fill_null(False), pl.col("NOISE").fill_null(False), pl.col("removed").fill_null(False))
                .with_columns(pl.when(pl.col("removed") | pl.col("DECOY")).then(0.0)
                                .when(pl.col("NOISE")).then(pl.max_horizontal(pl.col("p2"), pl.lit(P_NOISE)))
                                .otherwise(pl.col("p2")).alias("p2n")))
    new_sel = ev_select(cand.select("rid", "s1", pl.col("p2n").alias("p2")), "p2")
    keep = old.select("rid", "s1").join(aff, on="s1", how="anti")
    acc = pl.concat([keep, new_sel.select("rid", "s1")]).unique()
    print("pairs: in", old.height, "out", acc.height, "| affected S1", aff.height,
          "| in-file sel in affected", old.join(aff, on="s1").height, "new sel in affected", new_sel.height, flush=True)
    m = (s1id.join(acc.join(qid, on="rid").group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")), on="s1", how="left")
             .select("source1_entity_id", pl.col("matched_entity_ids").fill_null("")))
    m = sub.select("source1_entity_id").join(m, on="source1_entity_id", how="left", maintain_order="left")
    m.write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    x.filter(pl.col("DECOY") | pl.col("NOISE")).select("rid", "s1", "DECOY", "NOISE").write_parquet(f"{out_dir}/france_fix2_pairs.parquet")
    print("written", f"{out_dir}/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main()
