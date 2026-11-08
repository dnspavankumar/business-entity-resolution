"""Remove likely "decoy" matches from a matching_results.tsv.

What is a decoy?
    The data generator creates lookalike businesses next to real Source-1 entities: the same
    name with a changed legal suffix or an extra word, at the same street, but with the house
    number shifted by a fixed positive offset (+1, +2, +3, +4, +5, +7, +9, +11, +13 or +21).
    In the training data 79.8% of same-name US decoys carry one of these offsets, versus 0.126%
    of true US pairs (0.299% in India) for the strong offsets (+3..+21). True-match number noise
    is typo-like and symmetric (-1/-2 is as common as +1/+2); decoy offsets are always positive.

Rule (reproduces submission v4 from v2):
    For each predicted pair (S1, record), let d = house number(record) - house number(S1).
    Remove the pair if
      (a) d is a strong offset (+3,+4,+5,+7,+9,+11,+13,+21) and no other predicted record of the
          same S1 has the same d (true members that share an S1-level number shift usually come
          in groups; a lone shifted record is a decoy), or
      (b) the country has no training labels (e.g. France in the test set) and d is any decoy
          offset (+1..+21 list). Without labels for that country the model is least calibrated.

Only the provided train/test files are used (no external data).

Optional --extra (reproduces v5 from v2): after the rule above, remove further offset classes whose
decoy share was estimated at 42-75% (see EXTRA_CLASSES), but only in S1 rows where removing them
raises the expected per-S1 F0.5.

Optional --hidden (v6 = v5 + this): decoys whose shifted number is not the first number of the
address (common in Indian addresses: plot / door / survey numbers). A pair qualifies when exactly
one address number differs between S1 and record (new - old in the strong offset list) and the
name was changed. The decoy share of each (country, shared-number) cell is estimated at run time
with the symmetry test (count(+k) - count(-k)) / count(+k); cells below 0.5 are left alone, so
countries where the first-number rule already caught these decoys are not touched. Removal is again
gated by the expected per-S1 F0.5.

usage:
    python decoy_postfilter.py --matching in.tsv --dataset <student_resource/dataset> \
        --out out.tsv [--removed removed_pairs.tsv] [--extra] [--hidden]
"""
import argparse
import os

import polars as pl

STRONG = [3, 4, 5, 7, 9, 11, 13, 21]
WEAK = [1, 2]
HOUSE_NO = r"(?:^|[\s,#])0*(\d{1,6})(?:[A-Za-z]?\b|[-/ ])"


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def full_name(col: str) -> pl.Expr:
    return (pl.col(col).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def house_no(col: str) -> pl.Expr:
    return pl.col(col).str.extract(HOUSE_NO, 1).cast(pl.Int64, strict=False)


def explode_matches(m: pl.DataFrame) -> pl.DataFrame:
    m = m.rename({m.columns[0]: "s1", m.columns[1]: "ids"}).with_columns(pl.col("ids").fill_null(""))
    return (m.with_columns(pl.col("ids").str.split(",")).explode("ids")
             .filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).rename({"ids": "rid"}))


def f05(tp: int, fp: int, fn: int) -> float:
    """Per-entity F0.5 exactly as the scorer defines it (empty vs empty = 1.0)."""
    if tp == 0:
        return 1.0 if (fp == 0 and fn == 0) else 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 1.25 * prec * rec / (0.25 * prec + rec)


# Extra classes (v5). Decoy share q of each class was estimated on the test predictions with the
# symmetry of true-match number noise: true pairs with d=+k are as common as with d=-k, while
# decoys only produce +k, so decoys(class) ~= count(+k) - count(-k) within the same class.
EXTRA_CLASSES = [
    # (country, offsets, name exactly equal to S1?, S1 number confirmed by another member?, offset shared?, q)
    ("US", STRONG, False, True, True, 0.75),
    ("US", STRONG, False, False, True, 0.57),
    ("US", STRONG, True, False, True, 0.46),
    ("US", WEAK, False, True, False, 0.51),
    ("India", WEAK, False, True, False, 0.47),
    ("India", WEAK, False, False, False, 0.42),
]


def extra_rules(keep: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Remove more likely decoys, but only where the expected change in per-S1 F0.5 is positive."""
    from math import comb
    k = keep.drop("n_same_d")
    same = k.filter(pl.col("d").is_not_null()).group_by("s1", "d").len().rename({"len": "n_same_d"})
    grp = k.group_by("s1").agg((pl.col("d") == 0).sum().alias("n_eq"), pl.len().alias("m"))
    names = s1.select(pl.col("entity_id").alias("s1"), full_name("business_name").alias("f1"))
    k = (k.join(same, on=["s1", "d"], how="left").with_columns(pl.col("n_same_d").fill_null(0))
          .join(grp, on="s1").join(names, on="s1")
          .with_columns((full_name("business_name") == pl.col("f1")).alias("feq"),
                        (pl.col("n_eq") > 0).alias("conf"), (pl.col("n_same_d") >= 2).alias("shared"),
                        pl.lit(None, dtype=pl.Float64).alias("q")))
    for ctry, offs, feq, conf, shared, q in EXTRA_CLASSES:
        cond = ((pl.col("country") == ctry) & pl.col("d").is_in(offs) & (pl.col("feq") == feq)
                & (pl.col("conf") == conf) & (pl.col("shared") == shared) & pl.col("q").is_null())
        k = k.with_columns(pl.when(cond).then(pl.lit(q)).otherwise(pl.col("q")).alias("q"))
    cand = k.filter(pl.col("q").is_not_null())
    chosen = []
    for s1_id, qs, rids, m in cand.group_by("s1").agg(pl.col("q"), pl.col("rid"), pl.col("m").first()).iter_rows():
        r, qbar = len(qs), sum(qs) / len(qs)
        ev = sum(comb(r, j) * qbar ** j * (1 - qbar) ** (r - j) * (f05(m - r, 0, r - j) - f05(m - j, j, 0))
                 for j in range(r + 1))
        if ev > 0:
            chosen += [(s1_id, x) for x in rids]
    sel = pl.DataFrame(chosen, schema=["s1", "rid"], orient="row")
    return cand.join(sel, on=["s1", "rid"], how="semi").drop(["n_eq", "m", "f1", "feq", "conf", "shared", "q"])


def all_numbers(col: str) -> pl.Expr:
    return (pl.col(col).str.extract_all(r"\d+")
            .list.eval(pl.element().str.replace(r"^0+(\d)", "$1").cast(pl.Int64, strict=False)).list.unique())


def ev_select(cand: pl.DataFrame) -> pl.DataFrame:
    """cand has s1, rid, q, m. Keep the pairs of every S1 whose removal raises the expected F0.5."""
    from math import comb
    chosen = []
    for s1_id, qs, rids, m in cand.group_by("s1").agg(pl.col("q"), pl.col("rid"), pl.col("m").first()).iter_rows():
        r, qbar = len(qs), sum(qs) / len(qs)
        ev = sum(comb(r, j) * qbar ** j * (1 - qbar) ** (r - j) * (f05(m - r, 0, r - j) - f05(m - j, j, 0))
                 for j in range(r + 1))
        if ev > 0:
            chosen += [(s1_id, x) for x in rids]
    return pl.DataFrame(chosen, schema={"s1": cand.schema["s1"], "rid": cand.schema["rid"]}, orient="row")


def hidden_offset_rule(keep: pl.DataFrame, s1: pl.DataFrame, min_q: float = 0.5) -> pl.DataFrame:
    m = keep.group_by("s1").len().rename({"len": "m"})
    s1n = s1.select(pl.col("entity_id").alias("s1"), full_name("business_name").alias("f1"),
                    pl.col("business_address").alias("a1"))
    # name-changed pairs whose addresses both contain a digit (keeps memory low)
    k = (keep.select("s1", "rid", "country", "business_name", "business_address")
             .join(s1n, on="s1")
             .filter((full_name("business_name") != pl.col("f1"))
                     & pl.col("business_address").str.contains(r"\d") & pl.col("a1").str.contains(r"\d"))
             .select("s1", "rid", "country", all_numbers("business_address").alias("N"), all_numbers("a1").alias("N1"))
             .join(m, on="s1")
             .with_columns(pl.lit(False).alias("feq"))
             .with_columns(pl.col("N").list.set_difference(pl.col("N1")).alias("onlyR"),
                           pl.col("N1").list.set_difference(pl.col("N")).alias("onlyS"))
             .filter((pl.col("onlyR").list.len() == 1) & (pl.col("onlyS").list.len() == 1) & ~pl.col("feq"))
             .with_columns(pl.col("onlyR").list.first().alias("newnum"),
                           (pl.col("onlyR").list.first() - pl.col("onlyS").list.first()).alias("dd")))
    sh = k.group_by("s1", "newnum").len().rename({"len": "n_same_new"})
    k = k.join(sh, on=["s1", "newnum"]).with_columns((pl.col("n_same_new") >= 2).alias("shared"))
    cells = (k.group_by("country", "shared")
              .agg(pl.col("dd").is_in(STRONG).sum().alias("pos"), pl.col("dd").is_in([-x for x in STRONG]).sum().alias("neg"))
              .with_columns(((pl.col("pos").cast(pl.Int64) - pl.col("neg").cast(pl.Int64))
                             / pl.col("pos").cast(pl.Int64).clip(lower_bound=1)).alias("q")))
    print("hidden-offset cells (decoy share estimated by +k/-k symmetry):")
    print(cells.sort("country", "shared"))
    good = cells.filter(pl.col("q") >= min_q).select("country", "shared", "q")
    cand = k.filter(pl.col("dd").is_in(STRONG)).join(good, on=["country", "shared"])
    sel = ev_select(cand.select("s1", "rid", "q", "m"))
    return keep.join(sel, on=["s1", "rid"], how="semi")


def refit_rules(p: pl.DataFrame, s1: pl.DataFrame, min_q: float = 0.45, min_pos: int = 30) -> pl.DataFrame:
    """Data-driven version of the offset rules for ANY input matching file (used for v8 = v7 + refit).
    For every cell (country, strong/weak offset, exact-name?, S1-number confirmed?, offset shared?) the
    decoy share is estimated on the input itself by the +k/-k symmetry; cells with q >= min_q are
    removed where the expected per-S1 F0.5 rises."""
    eq = p.group_by("s1").agg((pl.col("d") == 0).sum().alias("n_eq"), pl.len().alias("m"))
    x = (p.join(eq, on="s1").join(s1.select(pl.col("entity_id").alias("s1"), full_name("business_name").alias("f1")), on="s1")
          .with_columns((full_name("business_name") == pl.col("f1")).alias("feq"), (pl.col("n_eq") > 0).alias("conf"),
                        (pl.col("n_same_d") >= 2).alias("shared")))
    rows = []
    for c in x["country"].unique().to_list():
        for gname, g in (("strong", STRONG), ("weak", WEAK)):
            for feq in (False, True):
                for conf in (False, True):
                    for sh in (False, True):
                        y = x.filter((pl.col("country") == c) & (pl.col("feq") == feq) & (pl.col("conf") == conf) & (pl.col("shared") == sh))
                        po = y.filter(pl.col("d").is_in(g)).height
                        ne = y.filter(pl.col("d").is_in([-k for k in g])).height
                        if po >= min_pos:
                            rows.append((c, gname, feq, conf, sh, po, ne, (po - ne) / po))
    cells = pl.DataFrame(rows, schema=["country", "grp", "feq", "conf", "shared", "pos", "neg", "q"], orient="row")
    print("refit cells with q >= %.2f:" % min_q)
    print(cells.filter(pl.col("q") >= min_q))
    cand = []
    for r in cells.filter(pl.col("q") >= min_q).iter_rows(named=True):
        g = STRONG if r["grp"] == "strong" else WEAK
        cand.append(x.filter((pl.col("country") == r["country"]) & (pl.col("feq") == r["feq"]) & (pl.col("conf") == r["conf"])
                             & (pl.col("shared") == r["shared"]) & pl.col("d").is_in(g)).with_columns(pl.lit(r["q"]).alias("q")))
    if not cand:
        return p.head(0)
    cand = pl.concat(cand)
    sel = ev_select(cand.select("s1", "rid", "q", "m"))
    return p.join(sel, on=["s1", "rid"], how="semi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--dataset", required=True, help="student_resource/dataset (has train/ and test/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--removed", default=None)
    ap.add_argument("--extra", action="store_true", help="also apply the v5 classes (expected-F0.5 gated)")
    ap.add_argument("--hidden", action="store_true", help="also remove decoys whose shifted number is not the first one (v6)")
    ap.add_argument("--refit", action="store_true",
                    help="skip the fixed v4/v5 rules; estimate every offset cell on the input by symmetry (v8 = v7 --refit --hidden)")
    a = ap.parse_args()

    test_dir, train_dir = os.path.join(a.dataset, "test"), os.path.join(a.dataset, "train")
    s1 = read_tsv(os.path.join(test_dir, "test_source1.tsv"))
    train_countries = set(read_tsv(os.path.join(train_dir, "train_source1.tsv"))["country"].unique().to_list())
    unlabeled = [c for c in s1["country"].unique().to_list() if c not in train_countries]
    print("countries without training labels (strict rule):", unlabeled)

    matching = read_tsv(a.matching)
    pairs = explode_matches(matching)
    recs = pl.concat([read_tsv(os.path.join(test_dir, f"test_source{s}.tsv")).select("entity_id", "business_name", "business_address")
                      for s in (2, 3)]).join(pairs.select(pl.col("rid").alias("entity_id")), on="entity_id", how="semi")
    p = (pairs
         .join(s1.select(pl.col("entity_id").alias("s1"), "country", pl.col("business_name").alias("s1_name"),
                         pl.col("business_address").alias("s1_address"), house_no("business_address").alias("h1")), on="s1")
         .join(recs.select(pl.col("entity_id").alias("rid"), "business_name", "business_address",
                           house_no("business_address").alias("h")), on="rid")
         .with_columns((pl.col("h") - pl.col("h1")).alias("d")))
    same = p.filter(pl.col("d").is_not_null()).group_by("s1", "d").len().rename({"len": "n_same_d"})
    p = p.join(same, on=["s1", "d"], how="left").with_columns(pl.col("n_same_d").fill_null(0))
    rule_a = pl.col("d").is_in(STRONG) & (pl.col("n_same_d") == 1)
    rule_b = pl.col("country").is_in(unlabeled) & pl.col("d").is_in(STRONG + WEAK)
    removed = refit_rules(p, s1) if a.refit else p.filter(rule_a | rule_b)
    print(f"pairs in: {p.height}, removed: {removed.height}")
    print(removed.group_by("country").len().sort("country"))

    keep = p.join(removed.select("s1", "rid"), on=["s1", "rid"], how="anti")
    if a.extra:
        extra = extra_rules(keep, s1)
        print(f"extra rules (v5): removed {extra.height} more pairs")
        print(extra.group_by("country").len().sort("country"))
        removed = pl.concat([removed, extra.select(removed.columns)])
        keep = keep.join(extra.select("s1", "rid"), on=["s1", "rid"], how="anti")
    if a.hidden:
        import gc
        del p, recs
        gc.collect()
        hid = hidden_offset_rule(keep, s1)
        print(f"hidden-offset rule (v6): removed {hid.height} more pairs")
        print(hid.group_by("country").len().sort("country"))
        removed = pl.concat([removed, hid.select(removed.columns)])
        keep = keep.join(hid.select("s1", "rid"), on=["s1", "rid"], how="anti")
    # keep the original order of ids inside each row
    order = pairs.with_row_index("pos").join(keep.select("s1", "rid"), on=["s1", "rid"], how="semi")
    lists = order.sort("pos").group_by("s1", maintain_order=True).agg(pl.col("rid").str.join(",").alias("ids"))
    out = (matching.select(pl.col(matching.columns[0]).alias("s1"))
           .join(lists, on="s1", how="left").with_columns(pl.col("ids").fill_null(""))
           .rename({"s1": "source1_entity_id", "ids": "matched_entity_ids"}))
    out.write_csv(a.out, separator="\t", quote_style="never")
    print("wrote", a.out, out.height, "rows")
    if a.removed:
        (removed.select(pl.col("s1").alias("source1_entity_id"), pl.col("rid").alias("removed_entity_id"), "country",
                        pl.col("d").alias("house_number_offset"), "s1_name", "s1_address",
                        pl.col("business_name").alias("removed_name"), pl.col("business_address").alias("removed_address"))
         .sort("source1_entity_id").write_csv(a.removed, separator="\t", quote_style="never"))


if __name__ == "__main__":
    main()
