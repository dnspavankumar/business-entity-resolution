"""Extra stage-2 features (v14 candidates).

1) Positional house-number comparison. anums keeps the numbers of an address in order ("5/5A" -> "5 5",
   "12/1/1" -> "12 1 1"). The decoy recipe shifts ONE number up by a small offset; for Indian addresses this is
   often not the first number ("5/5A" -> "5/7A"), which the set-based features in decoy_features.py miss.
2) Sibling agreement ("source profile"). The generator keeps one address variant per (entity, source): all
   Source-2 records of an entity share e.g. the same truncated house number or the same city spelling.
   A record whose address deviates from the Source-1 address in the same way as another confident record of the
   same entity from the same source is most likely a true record; a decoy has its own deviation.
Siblings of a best pair (r, x) = the other records whose best Source-1 candidate is x with stage-1 p1 >= 0.5.
All inputs exist for train (V/T best pairs) and test in the same way.
"""
import numpy as np
import polars as pl

STRONG = (3, 4, 5, 7, 9, 11, 13, 21)
WEAK = (1, 2)
SIB_MIN = 0.5
POS_FEATS = ["pn_len_r", "pn_len_s", "pn_ndiff", "pn_first_idx", "pn_first_d", "pn_first_strong", "pn_first_weak",
             "pn_first_negsmall", "pn_any_possmall", "pn_prefix_eq", "pn_last_d",
             # best alignment of the two number lists (shift k in -2..2), multiset leftovers
             "pa_k", "pa_eq", "pa_ndiff", "pa_any_possmall", "pa_any_strong", "pa_any_negsmall", "pa_min_absd",
             "ms_onlyR", "ms_onlyS", "ms_best_d", "ms_best_strong", "ms_best_possmall"]
SIB_FEATS = ["sib_ss_n", "sib_os_n", "ss_anums_eq", "os_anums_eq", "ss_atoks_eq", "os_atoks_eq", "ss_city_eq", "os_city_eq",
             "ss_ncore_eq", "os_ncore_eq", "anums_dev", "city_dev", "ss_anums_eq_dev", "os_anums_eq_dev", "ss_city_eq_dev",
             "ss_n1_eq_dev", "os_n1_eq_dev", "sib_agree_maxp1",
             # group counts over ALL best pairs of the entity (any p1): records sharing the same first number / numbers
             "g_ss_n1", "g_os_n1", "g_ss_anums", "g_os_anums", "g_ss_n1_dev", "g_os_n1_dev",
             # name-change signature (tokens added to / dropped from the Source-1 name) shared with other records of the entity
             "g_ss_addsig", "g_os_addsig", "g_ss_chgsig", "g_os_chgsig", "has_addsig"]
EXTRA_FEATS_ALL = POS_FEATS + SIB_FEATS
# The sibling / group features are NOT used: in the test, decoy entities often have 2+ records that share the
# shifted number (strong-offset pairs with a same-source record sharing the number: US 19% test vs 1.2% for train
# decoys), so "the deviation is shared" means true in train but not in the test. Only pair-level number features.
EXTRA_FEATS = POS_FEATS


def _pos(qm, sm):
    n = len(qm)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in POS_FEATS}
    for i in range(n):
        a = [int(x) for x in qm[i].split()] if qm[i] else []
        b = [int(x) for x in sm[i].split()] if sm[i] else []
        out["pn_len_r"][i] = len(a)
        out["pn_len_s"][i] = len(b)
        if not a or not b:
            continue
        m = min(len(a), len(b))
        diffs = [a[j] - b[j] for j in range(m)]
        nd = sum(1 for d in diffs if d != 0)
        out["pn_ndiff"][i] = nd
        pre = 0
        while pre < m and diffs[pre] == 0:
            pre += 1
        out["pn_prefix_eq"][i] = pre
        if nd:
            j = next(j for j in range(m) if diffs[j] != 0)
            d = diffs[j]
            out["pn_first_idx"][i] = j
            out["pn_first_d"][i] = max(-10000, min(10000, d))
            out["pn_first_strong"][i] = float(d in STRONG)
            out["pn_first_weak"][i] = float(d in WEAK)
            out["pn_first_negsmall"][i] = float(-21 <= d <= -1)
            out["pn_any_possmall"][i] = float(any(1 <= x <= 21 for x in diffs))
            out["pn_last_d"][i] = max(-10000, min(10000, diffs[max(k for k in range(m) if diffs[k] != 0)]))
        else:
            out["pn_first_idx"][i] = -1
            out["pn_first_d"][i] = 0
            out["pn_first_strong"][i] = 0
            out["pn_first_weak"][i] = 0
            out["pn_first_negsmall"][i] = 0
            out["pn_any_possmall"][i] = 0
            out["pn_last_d"][i] = 0
        # best alignment: a[j + k] vs b[j]
        bestk, beq, bd = 0, -1, []
        for k in (0, 1, -1, 2, -2):
            pairs = [(a[j + k], b[j]) for j in range(len(b)) if 0 <= j + k < len(a)]
            if not pairs:
                continue
            eq = sum(1 for u, v in pairs if u == v)
            if eq > beq:
                bestk, beq, bd = k, eq, [u - v for u, v in pairs if u != v]
        out["pa_k"][i] = bestk
        out["pa_eq"][i] = max(beq, 0)
        out["pa_ndiff"][i] = len(bd)
        out["pa_any_possmall"][i] = float(any(1 <= d <= 21 for d in bd))
        out["pa_any_strong"][i] = float(any(d in STRONG for d in bd))
        out["pa_any_negsmall"][i] = float(any(-21 <= d <= -1 for d in bd))
        out["pa_min_absd"][i] = min((abs(d) for d in bd), default=0)
        # multiset leftovers (duplicates kept), closest leftover pair
        ca, cb = {}, {}
        for u in a:
            ca[u] = ca.get(u, 0) + 1
        for v in b:
            cb[v] = cb.get(v, 0) + 1
        la = [u for u, c in ca.items() for _ in range(max(0, c - cb.get(u, 0)))]
        lb = [v for v, c in cb.items() for _ in range(max(0, c - ca.get(v, 0)))]
        out["ms_onlyR"][i] = len(la)
        out["ms_onlyS"][i] = len(lb)
        if la and lb:
            d = min((u - v for u in la for v in lb), key=abs)
            out["ms_best_d"][i] = max(-10000, min(10000, d))
            out["ms_best_strong"][i] = float(d in STRONG)
            out["ms_best_possmall"][i] = float(1 <= d <= 21)
    return out


def compute(best: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, pos_only: bool = True) -> pl.DataFrame:
    """best: rid, s1, p1 (all best pairs of complete clusters); q: rid, entity_id, anums, atoks, acity, ncore;
    s1: rid, anums, atoks, acity, ncore. Returns rid, s1 + EXTRA_FEATS."""
    tq = q.select("rid", pl.when(pl.col("entity_id").str.starts_with("S2")).then(2).otherwise(3).cast(pl.Int8).alias("src"),
                  *[pl.col(c).fill_null("") for c in ("anums", "atoks", "acity", "ncore")])
    ts = s1.select(pl.col("rid").alias("s1"), *[pl.col(c).fill_null("").alias("s_" + c) for c in ("anums", "atoks", "acity", "ncore")])
    x = best.select("rid", "s1", "p1").join(tq, on="rid", how="left").join(ts, on="s1", how="left")
    R, S = pl.col("ncore").str.split(" "), pl.col("s_ncore").str.split(" ")
    x = x.with_columns(R.list.set_difference(S).list.sort().list.join(" ").alias("addsig"),
                       S.list.set_difference(R).list.sort().list.join(" ").alias("dropsig"))
    x = x.with_columns(pl.concat_str([pl.col("addsig"), pl.lit("|"), pl.col("dropsig")]).alias("chgsig"))
    pos = _pos(x["anums"].fill_null("").to_list(), x["s_anums"].fill_null("").to_list())
    x = x.with_columns([pl.Series(k, v) for k, v in pos.items()])
    if pos_only:   # the features used by v14 (pair-level only)
        return x.select("rid", "s1", *POS_FEATS)
    x = x.with_columns(((pl.col("anums") != pl.col("s_anums")) & (pl.col("anums") != "") & (pl.col("s_anums") != "")).alias("anums_dev_b"),
                       ((pl.col("acity") != pl.col("s_acity")) & (pl.col("acity") != "") & (pl.col("s_acity") != "")).alias("city_dev_b"),
                       pl.col("anums").str.split(" ").list.first().fill_null("").alias("n1"),
                       pl.col("s_anums").str.split(" ").list.first().fill_null("").alias("s_n1"))
    sib = x.filter(pl.col("p1") >= SIB_MIN).select("s1", pl.col("rid").alias("rid2"), pl.col("src").alias("src2"), pl.col("p1").alias("p1b"),
                                                  pl.col("anums").alias("anums2"), pl.col("atoks").alias("atoks2"),
                                                  pl.col("acity").alias("acity2"), pl.col("ncore").alias("ncore2"), pl.col("n1").alias("n1b"))
    base = x.select("rid", "s1", "src", "anums", "atoks", "acity", "ncore", "n1", "s_n1", "anums_dev_b", "city_dev_b")
    aggs = []
    CH = 200_000
    for i in range(0, base.height, CH):
        pr = base.slice(i, CH).join(sib, on="s1").filter(pl.col("rid") != pl.col("rid2"))
        same = pl.col("src") == pl.col("src2")
        aeq = (pl.col("anums") == pl.col("anums2")) & (pl.col("anums") != "")
        teq = (pl.col("atoks") == pl.col("atoks2")) & (pl.col("atoks") != "")
        ceq = (pl.col("acity") == pl.col("acity2")) & (pl.col("acity") != "")
        neq = (pl.col("ncore") == pl.col("ncore2")) & (pl.col("ncore") != "")
        n1eq = (pl.col("n1") == pl.col("n1b")) & (pl.col("n1") != "") & (pl.col("n1") != pl.col("s_n1")) & (pl.col("s_n1") != "")
        aggs.append(pr.group_by("rid", "s1").agg(
            same.sum().alias("sib_ss_n"), (~same).sum().alias("sib_os_n"),
            (same & aeq).any().alias("ss_anums_eq"), (~same & aeq).any().alias("os_anums_eq"),
            (same & teq).any().alias("ss_atoks_eq"), (~same & teq).any().alias("os_atoks_eq"),
            (same & ceq).any().alias("ss_city_eq"), (~same & ceq).any().alias("os_city_eq"),
            (same & neq).any().alias("ss_ncore_eq"), (~same & neq).any().alias("os_ncore_eq"),
            (same & aeq & pl.col("anums_dev_b")).any().alias("ss_anums_eq_dev"),
            (~same & aeq & pl.col("anums_dev_b")).any().alias("os_anums_eq_dev"),
            (same & ceq & pl.col("city_dev_b")).any().alias("ss_city_eq_dev"),
            (same & n1eq).any().alias("ss_n1_eq_dev"), (~same & n1eq).any().alias("os_n1_eq_dev"),
            pl.when(aeq).then(pl.col("p1b")).otherwise(None).max().alias("sib_agree_maxp1")))
    agg = pl.concat(aggs) if aggs else None
    out = x.select("rid", "s1", *POS_FEATS, pl.col("anums_dev_b").cast(pl.Float32).alias("anums_dev"),
                   pl.col("city_dev_b").cast(pl.Float32).alias("city_dev"))
    if agg is not None:
        out = out.join(agg, on=["rid", "s1"], how="left")
    # group counts per (entity, number) and source, over all best pairs
    gn = x.filter(pl.col("n1") != "").group_by("s1", "n1").agg((pl.col("src") == 2).sum().alias("_g2"), (pl.col("src") == 3).sum().alias("_g3"))
    ga = x.filter(pl.col("anums") != "").group_by("s1", "anums").agg((pl.col("src") == 2).sum().alias("_a2"), (pl.col("src") == 3).sum().alias("_a3"))
    y = (x.select("rid", "s1", "src", "n1", "s_n1", "anums").join(gn, on=["s1", "n1"], how="left").join(ga, on=["s1", "anums"], how="left")
          .with_columns([pl.col(c).fill_null(0) for c in ("_g2", "_g3", "_a2", "_a3")]))
    two = pl.col("src") == 2
    dev1 = (pl.col("n1") != "") & (pl.col("s_n1") != "") & (pl.col("n1") != pl.col("s_n1"))
    y = y.select("rid", "s1",
                 pl.when(two).then(pl.col("_g2") - 1).otherwise(pl.col("_g3") - 1).clip(0, None).cast(pl.Float32).alias("g_ss_n1"),
                 pl.when(two).then(pl.col("_g3")).otherwise(pl.col("_g2")).cast(pl.Float32).alias("g_os_n1"),
                 pl.when(two).then(pl.col("_a2") - 1).otherwise(pl.col("_a3") - 1).clip(0, None).cast(pl.Float32).alias("g_ss_anums"),
                 pl.when(two).then(pl.col("_a3")).otherwise(pl.col("_a2")).cast(pl.Float32).alias("g_os_anums"),
                 pl.when(dev1).then(pl.when(two).then(pl.col("_g2") - 1).otherwise(pl.col("_g3") - 1).clip(0, None)).otherwise(-1).cast(pl.Float32).alias("g_ss_n1_dev"),
                 pl.when(dev1).then(pl.when(two).then(pl.col("_g3")).otherwise(pl.col("_g2"))).otherwise(-1).cast(pl.Float32).alias("g_os_n1_dev"))
    out = out.join(y, on=["rid", "s1"], how="left")
    gs = x.filter(pl.col("addsig") != "").group_by("s1", "addsig").agg((pl.col("src") == 2).sum().alias("_s2"), (pl.col("src") == 3).sum().alias("_s3"))
    gc_ = x.filter(pl.col("chgsig") != "|").group_by("s1", "chgsig").agg((pl.col("src") == 2).sum().alias("_c2"), (pl.col("src") == 3).sum().alias("_c3"))
    z = (x.select("rid", "s1", "src", "addsig", "chgsig").join(gs, on=["s1", "addsig"], how="left").join(gc_, on=["s1", "chgsig"], how="left")
          .with_columns([pl.col(c).fill_null(0) for c in ("_s2", "_s3", "_c2", "_c3")]))
    has = pl.col("addsig") != ""
    hasc = pl.col("chgsig") != "|"
    z = z.select("rid", "s1",
                 pl.when(has).then(pl.when(two).then(pl.col("_s2") - 1).otherwise(pl.col("_s3") - 1).clip(0, None)).otherwise(-1).cast(pl.Float32).alias("g_ss_addsig"),
                 pl.when(has).then(pl.when(two).then(pl.col("_s3")).otherwise(pl.col("_s2"))).otherwise(-1).cast(pl.Float32).alias("g_os_addsig"),
                 pl.when(hasc).then(pl.when(two).then(pl.col("_c2") - 1).otherwise(pl.col("_c3") - 1).clip(0, None)).otherwise(-1).cast(pl.Float32).alias("g_ss_chgsig"),
                 pl.when(hasc).then(pl.when(two).then(pl.col("_c3")).otherwise(pl.col("_c2"))).otherwise(-1).cast(pl.Float32).alias("g_os_chgsig"),
                 has.cast(pl.Float32).alias("has_addsig"))
    out = out.join(z, on=["rid", "s1"], how="left")
    for c in SIB_FEATS:
        if c in ("anums_dev", "city_dev", "has_addsig") or c.startswith("g_"):
            continue
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))
        fill = -1.0 if c == "sib_agree_maxp1" else 0.0
        out = out.with_columns(pl.col(c).cast(pl.Float32).fill_null(fill))
    return out.select("rid", "s1", *EXTRA_FEATS)
