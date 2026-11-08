"""Decoy-vs-noise features for (query record, Source-1) pairs.

The data generator produces two kinds of lookalikes of a Source-1 entity:
  * TRUE variants  = noise(S1): typos, case/accents/leet, abbreviations, legal-suffix drop or
    expansion, word re-ordering or dropping, made-up alias names ("Halobelo", "X a/k/a Y"),
    domain-style names, initials, native script; addresses with missing parts, zero padding,
    digit drops and small number typos in BOTH directions (-2..+2).
  * DECOYS         = noise(decoy_op(S1)): the legal suffix swapped or added, a real business
    word added or substituted ("Industries", "Enterprises", "Group", "Services", "France",
    "Développement", ...), and one address number shifted UP by +1,+2,+3,+4,+5,+7,+9,+11,+13 or +21
    (or the same name at another location, or another business at the same address).
    Decoys can have several noisy copies of their own.
The main pipeline only sees "numbers equal / not equal" and similarity scores, so it cannot tell
these apart. The features below describe *which operations* turn the S1 record into the query.

All inputs come from the normalised parquet files written by prep.py (columns ncore, nfull,
nalt, atoks, anums, acity, state, is_domain, is_dba ...). No external data.
"""
import math
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

STRONG = (3, 4, 5, 7, 9, 11, 13, 21)
WEAK = (1, 2)
OFFSETS = set(STRONG) | set(WEAK)
LEGAL = {"pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pc", "pllc", "sarl", "sas", "sasu",
         "eurl", "sa", "ei", "sci", "snc", "scop", "selarl", "opc", "limited", "private", "corporation",
         "incorporated", "company"}

FEATURE_NAMES = [
    # numbers
    "d1", "d1_strong", "d1_weak", "d1_negsmall", "d1_possmall_other", "d1_digitdrop", "d1_ham1",
    "n_onlyR", "n_onlyS", "one_repl", "dd", "dd_strong", "dd_weak", "dd_neg", "any_pos_strong", "any_neg_strong",
    # names
    "n_same", "n_typo", "n_add", "n_del", "n_add_vocab", "n_add_pseudo", "add_min_df", "add_max_df",
    "n_sub_vocab", "legal_added", "legal_removed", "legal_swapped", "q_novel_frac", "q_all_novel",
    "initials_match", "name_exact_full", "name_exact_core", "first_tok_typo", "max_typo_dist",
    # address words
    "street_eq", "street_jac", "city_overlap",
]


def _nums(s):
    return [int(x) for x in s.split()] if s else []


def _pair_feats(args):
    """args: lists for a chunk. Returns dict of numpy arrays."""
    (qn, sn, qf, sf, qa, sa, qm, sm, qc, sc, vocab) = args
    n = len(qn)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in FEATURE_NAMES}
    for i in range(n):
        # ---------------- numbers
        a, b = _nums(qm[i]), _nums(sm[i])
        if a and b:
            d1 = a[0] - b[0]
            out["d1"][i] = max(-10000, min(10000, d1))
            out["d1_strong"][i] = float(d1 in STRONG)
            out["d1_weak"][i] = float(d1 in WEAK)
            out["d1_negsmall"][i] = float(-21 <= d1 <= -1)
            out["d1_possmall_other"][i] = float(1 <= d1 <= 21 and d1 not in OFFSETS)
            x, y = str(a[0]), str(b[0])
            out["d1_digitdrop"][i] = float(x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x)))
            out["d1_ham1"][i] = float(len(x) == len(y) and sum(c1 != c2 for c1, c2 in zip(x, y)) == 1)
            sa_, sb_ = set(a), set(b)
            only_r, only_s = sa_ - sb_, sb_ - sa_
            out["n_onlyR"][i] = len(only_r)
            out["n_onlyS"][i] = len(only_s)
            one = len(only_r) == 1 and len(only_s) == 1
            out["one_repl"][i] = float(one)
            if one:
                dd = next(iter(only_r)) - next(iter(only_s))
                out["dd"][i] = max(-10000, min(10000, dd))
                out["dd_strong"][i] = float(dd in STRONG)
                out["dd_weak"][i] = float(dd in WEAK)
                out["dd_neg"][i] = float(-21 <= dd <= -1)
            if only_r and only_s:
                diffs = {r - s for r in only_r for s in only_s}
                out["any_pos_strong"][i] = float(any(d in STRONG for d in diffs))
                out["any_neg_strong"][i] = float(any(-d in STRONG for d in diffs))
        # ---------------- names (core tokens, legal words from full tokens)
        A = sn[i].split() if sn[i] else []
        B = qn[i].split() if qn[i] else []
        FA = sf[i].split() if sf[i] else []
        FB = qf[i].split() if qf[i] else []
        la = {t for t in FA if t in LEGAL}
        lb = {t for t in FB if t in LEGAL}
        out["legal_added"][i] = float(bool(lb - la))
        out["legal_removed"][i] = float(bool(la - lb))
        out["legal_swapped"][i] = float(bool(lb - la) and bool(la - lb))
        out["name_exact_full"][i] = float(qf[i] == sf[i])
        out["name_exact_core"][i] = float(qn[i] == sn[i])
        setA = set(A)
        same = [t for t in B if t in setA]
        restA = [t for t in A if t not in set(B)]
        restB = [t for t in B if t not in setA]
        n_typo = 0
        max_td = 0
        used = set()
        adds = []
        for t in restB:
            best, bj = 0, -1
            for j, u in enumerate(restA):
                if j in used:
                    continue
                r = fuzz.ratio(t, u)
                if r > best:
                    best, bj = r, j
            if bj >= 0 and (best >= 75 or (len(t) >= 3 and (restA[bj].startswith(t) or t.startswith(restA[bj])))):
                used.add(bj)
                n_typo += 1
                max_td = max(max_td, Levenshtein.distance(t, restA[bj]))
            else:
                adds.append(t)
        n_del = len(restA) - len(used)
        out["n_same"][i] = len(same)
        out["n_typo"][i] = n_typo
        out["n_add"][i] = len(adds)
        out["n_del"][i] = n_del
        out["max_typo_dist"][i] = max_td
        dfs = [vocab.get(t, 0) for t in adds]
        out["n_add_vocab"][i] = sum(1 for x in dfs if x >= 3)
        out["n_add_pseudo"][i] = sum(1 for x in dfs if x < 3)
        out["add_min_df"][i] = math.log1p(min(dfs)) if dfs else -1.0
        out["add_max_df"][i] = math.log1p(max(dfs)) if dfs else -1.0
        out["n_sub_vocab"][i] = float(min(n_del, sum(1 for x in dfs if x >= 3)))
        if B:
            nov = [t for t in B if vocab.get(t, 0) < 3 and t not in setA]
            out["q_novel_frac"][i] = len(nov) / len(B)
            out["q_all_novel"][i] = float(len(nov) == len(B))
            if A:
                out["first_tok_typo"][i] = float(B[0] != A[0] and fuzz.ratio(B[0], A[0]) >= 75)
        if len(B) == 1 and len(A) >= 2:
            ini = "".join(t[0] for t in FA if t and t not in {"and", "the", "of"})
            ini2 = "".join(t[0] for t in A)
            out["initials_match"][i] = float(B[0] in (ini, ini2) or (len(B[0]) >= 2 and ini.startswith(B[0])))
        # ---------------- address words (non-numeric tokens, city components)
        wa = {t for t in (qa[i].split() if qa[i] else []) if not t.isdigit()}
        wb = {t for t in (sa[i].split() if sa[i] else []) if not t.isdigit()}
        ca = set(qc[i].split("|")) if qc[i] else set()
        cb = set(sc[i].split("|")) if sc[i] else set()
        wa2, wb2 = wa - {w for c in ca for w in c.split()}, wb - {w for c in cb for w in c.split()}
        if wa2 and wb2:
            out["street_eq"][i] = float(wa2 == wb2)
            out["street_jac"][i] = len(wa2 & wb2) / len(wa2 | wb2)
        if ca and cb:
            out["city_overlap"][i] = float(bool(ca & cb))
    return out


def _op_tokens_chunk(args):
    """For each pair: tokens ADDED in the record (not in S1, not a typo of an S1 token) and S1 tokens DROPPED."""
    qn, sn = args
    added, dropped = [], []
    for a_s, b_s in zip(sn, qn):
        A = a_s.split() if a_s else []
        B = b_s.split() if b_s else []
        setA, setB = set(A), set(B)
        restA = [t for t in A if t not in setB]
        restB = [t for t in B if t not in setA]
        used, adds = set(), []
        for t in restB:
            best, bj = 0, -1
            for j, u in enumerate(restA):
                if j in used:
                    continue
                r = fuzz.ratio(t, u)
                if r > best:
                    best, bj = r, j
            if bj >= 0 and (best >= 75 or (len(t) >= 3 and (restA[bj].startswith(t) or t.startswith(restA[bj])))):
                used.add(bj)
            else:
                adds.append(t)
        added.append(adds)
        dropped.append([u for j, u in enumerate(restA) if j not in used])
    return added, dropped


def op_tokens(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, nproc: int = 2, chunk: int = 200_000) -> pl.DataFrame:
    """pairs: rid, s1 -> rid, s1, added (list[str]), dropped (list[str]) on the core name tokens."""
    x = (pairs.select("rid", "s1").join(q.select("rid", "ncore"), on="rid", how="left")
              .join(s1.select(pl.col("rid").alias("s1"), pl.col("ncore").alias("s_ncore")), on="s1", how="left")
              .with_columns(pl.col("ncore").fill_null(""), pl.col("s_ncore").fill_null("")))
    starts = list(range(0, x.height, chunk))
    added, dropped = [], []
    with Pool(nproc) as p:
        for i in range(0, len(starts), nproc):
            batch = [(x.slice(s0, chunk)["ncore"].to_list(), x.slice(s0, chunk)["s_ncore"].to_list()) for s0 in starts[i:i + nproc]]
            for a, d in p.map(_op_tokens_chunk, batch):
                added += a
                dropped += d
    return x.select("rid", "s1").with_columns(pl.Series("added", added, dtype=pl.List(pl.Utf8)),
                                              pl.Series("dropped", dropped, dtype=pl.List(pl.Utf8)))


def decoy_word_scores(ops: pl.DataFrame, weak: pl.DataFrame, alpha: float = 1.0, min_count: int = 5) -> pl.DataFrame:
    """Self-supervised 'decoy word' score per (country, token), learned WITHOUT labels.
    ops: rid, s1, added ; weak: rid, s1, country, weak_decoy (bool: shifted-number fingerprint),
    weak_true (bool: same house number). score = log-odds of a token being added in weak-decoy vs weak-true pairs."""
    x = ops.join(weak, on=["rid", "s1"]).explode("added").filter(pl.col("added").is_not_null())
    cnt = x.group_by("country", "added").agg(pl.col("weak_decoy").sum().alias("cd"), pl.col("weak_true").sum().alias("ct"))
    tot = weak.group_by("country").agg(pl.col("weak_decoy").sum().alias("Nd"), pl.col("weak_true").sum().alias("Nt"))
    cnt = cnt.join(tot, on="country").filter((pl.col("cd") + pl.col("ct")) >= min_count)
    return cnt.with_columns(((pl.col("cd") + alpha) / (pl.col("Nd") + alpha)).log().alias("ld"),
                            ((pl.col("ct") + alpha) / (pl.col("Nt") + alpha)).log().alias("lt")).with_columns(
        (pl.col("ld") - pl.col("lt")).alias("dw_score")).select("country", pl.col("added").alias("tok"), "dw_score", "cd", "ct")


def decoy_word_features(ops: pl.DataFrame, scores: pl.DataFrame, country: pl.DataFrame) -> pl.DataFrame:
    """ops: rid, s1, added ; country: rid, s1, country -> max / sum of decoy-word scores of the added tokens."""
    x = (ops.join(country, on=["rid", "s1"]).explode("added")
            .join(scores.select("country", pl.col("tok").alias("added"), "dw_score"), on=["country", "added"], how="left"))
    agg = x.group_by("rid", "s1").agg(pl.col("dw_score").max().alias("dw_max"), pl.col("dw_score").fill_null(0).sum().alias("dw_sum"),
                                      (pl.col("dw_score") > 1.0).sum().alias("dw_n_hi"))
    return ops.select("rid", "s1").join(agg, on=["rid", "s1"], how="left").with_columns(
        pl.col("dw_max").fill_null(-9.0), pl.col("dw_sum").fill_null(0.0), pl.col("dw_n_hi").fill_null(0))


def s1_vocab(s1: pl.DataFrame) -> dict:
    """document frequency of core name tokens over Source-1 names (the 'real word' vocabulary)."""
    v = (s1.select(pl.col("ncore").str.split(" ").list.unique().alias("t")).explode("t")
           .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).group_by("t").len())
    return dict(zip(v["t"].to_list(), v["len"].to_list()))


def compute(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, vocab: dict, nproc: int = 2,
            chunk: int = 200_000) -> pl.DataFrame:
    """pairs: rid, s1 (s1 = Source-1 row id). q / s1 carry rid + ncore, nfull, atoks, anums, acity."""
    cols = ["ncore", "nfull", "atoks", "anums", "acity"]
    x = (pairs.select("rid", "s1")
              .join(q.select(["rid"] + cols), on="rid", how="left")
              .join(s1.select([pl.col("rid").alias("s1")] + [pl.col(c).alias("s_" + c) for c in cols]), on="s1", how="left"))
    x = x.with_columns([pl.col(c).fill_null("") for c in cols + ["s_" + c for c in cols]])
    def job(c):
        return (c["ncore"].to_list(), c["s_ncore"].to_list(), c["nfull"].to_list(), c["s_nfull"].to_list(),
                c["atoks"].to_list(), c["s_atoks"].to_list(), c["anums"].to_list(), c["s_anums"].to_list(),
                c["acity"].to_list(), c["s_acity"].to_list(), vocab)
    parts = []
    starts = list(range(0, x.height, chunk))
    if nproc > 1:
        with Pool(nproc) as p:
            for i in range(0, len(starts), nproc):   # bounded: only nproc chunks in memory at a time
                batch = [job(x.slice(s0, chunk)) for s0 in starts[i:i + nproc]]
                for r in p.map(_pair_feats, batch):
                    parts.append(pl.DataFrame(r))
                del batch
    else:
        for s0 in starts:
            parts.append(pl.DataFrame(_pair_feats(job(x.slice(s0, chunk)))))
    f = pl.concat(parts)
    return pl.concat([x.select("rid", "s1"), f], how="horizontal")


def cluster_features(best: pl.DataFrame, feats: pl.DataFrame, q: pl.DataFrame, member_min: float = 0.3) -> pl.DataFrame:
    """Agreement of each query with the other members of its best S1's cluster.
    best: rid, s1, p1 ; feats: rid, s1, d1 ; q: rid, anums, nfull."""
    b = (best.select("rid", "s1", "p1").join(feats.select("rid", "s1", "d1"), on=["rid", "s1"], how="left")
             .join(q.select("rid", pl.col("anums").str.split(" ").list.first().alias("h"), "nfull"), on="rid", how="left"))
    mem = b.filter(pl.col("p1") >= member_min)
    # how many (other) members have exactly the S1's number (d1 == 0) -> the S1 number is confirmed
    eq = mem.group_by("s1").agg((pl.col("d1") == 0).sum().alias("m_eq"), pl.len().alias("m_n"))
    same_d = mem.filter(pl.col("d1").is_not_null()).group_by("s1", "d1").len().rename({"len": "m_same_d"})
    same_name = mem.group_by("s1", "nfull").len().rename({"len": "m_same_name"})
    b = (b.join(eq, on="s1", how="left").join(same_d, on=["s1", "d1"], how="left").join(same_name, on=["s1", "nfull"], how="left")
          .with_columns([pl.col(c).fill_null(0) for c in ("m_eq", "m_n", "m_same_d", "m_same_name")]))
    self_in = (pl.col("p1") >= member_min).cast(pl.Int64)
    b = b.with_columns(
        (pl.col("m_eq") - (self_in * (pl.col("d1") == 0).fill_null(False).cast(pl.Int64))).alias("ctx_s1num_confirmed"),
        (pl.col("m_same_d") - self_in).clip(lower_bound=0).alias("ctx_same_offset"),
        (pl.col("m_same_name") - self_in).clip(lower_bound=0).alias("ctx_same_name"),
        (pl.col("m_n") - self_in).alias("ctx_n"),
    )
    return b.select("rid", "s1", "ctx_s1num_confirmed", "ctx_same_offset", "ctx_same_name", "ctx_n")
