"""Rescue search for records that the main blocking gave NO candidate at all.

Most of these records are true matches with heavy noise (train: 84-96% of them are true, except US
records with an address, 49%): typos in the name ("Surgical Sptcalists Inc"), web-address names
("chennaiprime.com"), invented brand names ("Miravera") whose only clue is the address, or addresses
with typos and no house number ("BEECH ST, EAON CDP, OH").

For each such record we search the Source-1 entities of the same country with two TF-IDF indexes:
  * name: character 3-grams of the normalised full name without spaces (typo and concatenation tolerant)
  * address: character 3-grams of the normalised address tokens + numbers + state
and keep the top K entities of each (cosine similarity). Very common 3-grams are dropped (max_df).

usage: python rescue_cands.py <work_dir> <split: train|test> <queries.parquet (rid)> <out.parquet>
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

K = 30
MAX_DF = 3000
CHUNK = 200
t0 = time.time()


def log(*a):
    print(*a, f"{time.time() - t0:.0f}s", flush=True)


def name_str(df):
    return df.select(pl.col("nfull").fill_null("").str.replace_all(" ", "")).to_series().to_list()


def addr_str(df):
    return df.select(pl.concat_str([pl.col("atoks").fill_null(""), pl.col("anums").fill_null(""), pl.col("state").fill_null("")],
                                   separator=" ").str.replace_all(r"\|", " ")).to_series().to_list()


def topk(Q, S, k, min_sim):
    """rows of Q (queries) against rows of S (entities); both L2-normalised sparse. returns (qi, si, sim)."""
    St = S.T.tocsc() if False else S.T.tocsr()
    out_q, out_s, out_v = [], [], []
    for a in range(0, Q.shape[0], CHUNK):
        M = (Q[a:a + CHUNK] @ St).tocsr()
        for i in range(M.shape[0]):
            s, e = M.indptr[i], M.indptr[i + 1]
            if e == s:
                continue
            v = M.data[s:e]; idx = M.indices[s:e]
            if len(v) > k:
                p = np.argpartition(-v, k)[:k]
                v, idx = v[p], idx[p]
            m = v >= min_sim
            out_q.append(np.full(m.sum(), a + i, dtype=np.int64)); out_s.append(idx[m]); out_v.append(v[m])
    if not out_q:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(out_q), np.concatenate(out_s), np.concatenate(out_v)


def search(q, s, k, max_df):
    """name and address TF-IDF search of queries q against entities s; returns rid, s1, name_sim, addr_sim"""
    if q.height == 0 or s.height == 0:
        return None
    vn = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_df=max_df, sublinear_tf=True, dtype=np.float32)
    try:
        Sn = vn.fit_transform(name_str(s))
    except ValueError:
        return None
    Qn = vn.transform(name_str(q))
    qi, si, _ = topk(Qn, Sn, k, 0.2)
    parts = [pl.DataFrame({"qi": qi, "si": si})]
    va = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_df=max_df, sublinear_tf=True, dtype=np.float32)
    try:
        Sa = va.fit_transform(addr_str(s)); Qa = va.transform(addr_str(q))
        qne = np.nonzero(~q["ea"].to_numpy())[0]
        if len(qne):
            qi2, si2, _ = topk(Qa[qne], Sa, k, 0.2)
            parts.append(pl.DataFrame({"qi": qne[qi2] if len(qi2) else qi2, "si": si2}))
    except ValueError:
        Sa = Qa = None
    cand = pl.concat(parts).unique()
    if cand.height == 0:
        return None
    qn_i = cand["qi"].to_numpy(); sn_i = cand["si"].to_numpy()
    name_sim = np.asarray(Qn[qn_i].multiply(Sn[sn_i]).sum(axis=1)).ravel()
    addr_sim = np.asarray(Qa[qn_i].multiply(Sa[sn_i]).sum(axis=1)).ravel() if Sa is not None else np.zeros(len(qn_i))
    return pl.DataFrame({"rid": q["rid"].to_numpy()[qn_i], "s1": s["rid"].to_numpy()[sn_i],
                         "name_sim": name_sim.astype(np.float32), "addr_sim": addr_sim.astype(np.float32)})


def main():
    W, split, qfile, out = sys.argv[1:5]
    cols = ["rid", "country", "nfull", "atoks", "anums", "state"]
    qids = pl.read_parquet(qfile).select("rid")
    q_all = pl.read_parquet(f"{W}/{split}_q.parquet", columns=cols + ["business_address"]).join(qids, on="rid")
    q_all = q_all.with_columns((pl.col("business_address").fill_null("").str.strip_chars() == "").alias("ea"),
                               pl.col("state").fill_null("")).drop("business_address")
    s1_all = pl.read_parquet(f"{W}/{split}_s1.parquet", columns=cols).with_columns(pl.col("state").fill_null(""))
    res = []
    for c in q_all["country"].unique().sort().to_list():
        q = q_all.filter(pl.col("country") == c)
        s = s1_all.filter(pl.col("country") == c)
        log(split, c, "queries", q.height, "entities", s.height)
        # 1) records with an address: search inside the same state (name + address), no document-frequency cap needed
        qs = q.filter(~pl.col("ea") & (pl.col("state") != ""))
        n0 = 0
        for st, qq in qs.group_by("state"):
            r = search(qq, s.filter(pl.col("state") == st[0]), K, 1.0)
            if r is not None:
                res.append(r); n0 += r.height
        log(" in-state candidates", n0)
        # 2) empty address / no state: name search over the whole country
        qe = q.filter(pl.col("ea") | (pl.col("state") == ""))
        r = search(qe.with_columns(pl.lit(True).alias("ea")), s, 2 * K, MAX_DF)
        if r is not None:
            res.append(r)
        log(" country-wide candidates", 0 if r is None else r.height)
    pl.concat(res).unique(["rid", "s1"]).write_parquet(out)
    log("written", out)


if __name__ == "__main__":
    main()
