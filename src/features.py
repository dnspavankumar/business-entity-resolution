"""Pairwise feature computation for (query = Source2/3 record, candidate = Source1 record)."""
import numpy as np
import polars as pl
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler

LEGAL = ["pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pc", "pllc", "sarl", "sas", "sasu",
         "eurl", "sa", "ei", "sci", "snc", "scop", "selarl", "opc"]
REC_COLS = ["rid", "country", "nfull", "ncore", "nalt", "is_domain", "is_dba", "is_native", "atoks", "anums",
            "state", "pin", "acity"]


def token_idf(frames, col, country_col="country"):
    """IDF table (country, tok, idf) from a list of lazy frames holding a space separated token column."""
    parts = [f.select(country_col, pl.col(col).str.split(" ").list.unique().alias("tok")) for f in frames]
    t = pl.concat(parts).explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    df = t.group_by([country_col, "tok"]).agg(pl.len().alias("df")).collect(engine="streaming")
    n = pl.concat([f.group_by(country_col).agg(pl.len().alias("n")) for f in frames]).collect(engine="streaming").group_by(country_col).agg(pl.col("n").sum())
    df = df.join(n, on=country_col).with_columns((pl.col("n").cast(pl.Float64) / pl.col("df")).log().cast(pl.Float32).alias("idf"))
    return df.select(country_col, "tok", "idf", "df")


def _cp(a, b, scorer, workers=2):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _weighted_overlap(p: pl.DataFrame, qcol: str, scol: str, idf: pl.DataFrame, prefix: str):
    """IDF-weighted overlap of token sets. p has pid, country, qcol, scol (space separated strings)."""
    base = p.select("pid", "country", pl.col(qcol).str.split(" ").list.unique().alias("qt"), pl.col(scol).str.split(" ").list.unique().alias("st"))
    qx = base.select("pid", "country", "st", pl.col("qt").alias("tok")).explode("tok").filter(pl.col("tok") != "")
    qx = qx.join(idf, on=["country", "tok"], how="left").with_columns(pl.col("idf").fill_null(12.0))
    qx = qx.with_columns(pl.col("st").list.contains(pl.col("tok")).alias("hit"))
    qa = qx.group_by("pid").agg(pl.col("idf").sum().alias("qw"), (pl.col("idf") * pl.col("hit")).sum().alias("hw"),
                                pl.col("idf").filter(~pl.col("hit")).max().alias("q_miss_max"))
    sx = base.select("pid", "country", pl.col("st").alias("tok")).explode("tok").filter(pl.col("tok") != "")
    sx = sx.join(idf, on=["country", "tok"], how="left").with_columns(pl.col("idf").fill_null(12.0))
    sa = sx.group_by("pid").agg(pl.col("idf").sum().alias("sw"))
    r = p.select("pid").join(qa, on="pid", how="left").join(sa, on="pid", how="left")
    r = r.with_columns([pl.col(c).fill_null(0.0) for c in ("qw", "hw", "sw", "q_miss_max")])
    return r.select("pid",
                    (pl.col("hw") / pl.col("qw").clip(lower_bound=1e-6)).alias(f"{prefix}_wq"),
                    (pl.col("hw") / pl.col("sw").clip(lower_bound=1e-6)).alias(f"{prefix}_ws"),
                    (pl.col("hw") / (pl.col("qw") + pl.col("sw") - pl.col("hw")).clip(lower_bound=1e-6)).alias(f"{prefix}_wj"),
                    pl.col("hw").alias(f"{prefix}_hw"), pl.col("q_miss_max").alias(f"{prefix}_qmiss"))


BREL_MIN = 0.3


def add_rank_feats(df: pl.DataFrame) -> pl.DataFrame:
    """Deterministic blocking-rank features (ties share a rank)."""
    return df.with_columns(
        pl.col("bscore").rank("min", descending=True).over("rid").cast(pl.Float32).alias("brank_min"),
        (pl.col("bscore") == pl.col("btop")).sum().over("rid").cast(pl.Float32).alias("n_top_ties"),
    )


def compute_features(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, name_idf: pl.DataFrame, addr_idf: pl.DataFrame):
    """pairs: rid, s1, bscore, nkeys, brank, btop (+ anything else, kept).  q/s1: REC_COLS tables."""
    pairs = pairs.filter(pl.col("bscore") >= BREL_MIN * pl.col("btop"))
    p = pairs.with_row_index("pid")
    qq = q.select(REC_COLS).rename({c: f"q_{c}" for c in REC_COLS if c != "rid"})
    ss = s1.select(REC_COLS).rename({c: f"s_{c}" for c in REC_COLS if c != "rid"}).rename({"rid": "s1"})
    p = p.join(qq, on="rid", how="left").join(ss, on="s1", how="left").sort("pid")
    p = p.with_columns(pl.col("q_country").alias("country"))
    F = {}
    qn, sn = p["q_ncore"].to_list(), p["s_ncore"].to_list()
    qf, sf = p["q_nfull"].to_list(), p["s_nfull"].to_list()
    qa, sa = p["q_atoks"].to_list(), p["s_atoks"].to_list()
    qalt = p["q_nalt"].to_list()
    qcc = [x.replace(" ", "") for x in qn]; scc = [x.replace(" ", "") for x in sn]
    F["n_ratio"] = _cp(qn, sn, fuzz.ratio)
    F["n_tsort"] = _cp(qn, sn, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(qn, sn, fuzz.token_set_ratio)
    F["n_partial"] = _cp(qn, sn, fuzz.partial_ratio)
    F["nf_tsort"] = _cp(qf, sf, fuzz.token_sort_ratio)
    F["nf_tset"] = _cp(qf, sf, fuzz.token_set_ratio)
    F["cc_ratio"] = _cp(qcc, scc, fuzz.ratio)
    F["cc_partial"] = _cp(qcc, scc, fuzz.partial_ratio)
    F["cc_jw"] = _cp(qcc, scc, JaroWinkler.normalized_similarity)
    alt_mask = np.array([1 if x else 0 for x in qalt], dtype=np.int8)
    F["alt_tset"] = np.where(alt_mask == 1, _cp(qalt, sn, fuzz.token_set_ratio), -1).astype(np.float32)
    F["a_ratio"] = _cp(qa, sa, fuzz.ratio)
    F["a_tsort"] = _cp(qa, sa, fuzz.token_sort_ratio)
    F["a_tset"] = _cp(qa, sa, fuzz.token_set_ratio)
    F["a_partial"] = _cp(qa, sa, fuzz.partial_ratio)
    qc_, sc_ = p["q_acity"].to_list(), p["s_acity"].to_list()
    F["city_tset"] = _cp([x.replace("|", " ") for x in qc_], [x.replace("|", " ") for x in sc_], fuzz.token_set_ratio)
    # last city component (usually the city)
    F["city_last"] = _cp([x.split("|")[-1] if x else "" for x in qc_], [x.split("|")[-1] if x else "" for x in sc_], fuzz.ratio)
    del qn, sn, qf, sf, qa, sa, qalt, qcc, scc, qc_, sc_
    feats = pl.DataFrame({k: v for k, v in F.items()})
    del F
    # --- set based features (polars) ---
    sp = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    s = p.select(
        sp("q_ncore").alias("qn"), sp("s_ncore").alias("sn"), sp("q_nfull").alias("qf"), sp("s_nfull").alias("sf"),
        sp("q_atoks").alias("qa"), sp("s_atoks").alias("sa"), sp("q_anums").alias("qm"), sp("s_anums").alias("sm"),
        pl.col("q_anums").str.split(" ").list.first().alias("qm1"), pl.col("s_anums").str.split(" ").list.first().alias("sm1"),
        pl.col("q_state"), pl.col("s_state"), pl.col("q_pin"), pl.col("s_pin"),
        pl.col("q_ncore").str.split(" ").list.first().alias("qn1"), pl.col("s_ncore").str.split(" ").list.first().alias("sn1"),
    )
    s = s.with_columns(
        pl.col("qf").list.eval(pl.element().filter(pl.element().is_in(LEGAL))).alias("ql"),
        pl.col("sf").list.eval(pl.element().filter(pl.element().is_in(LEGAL))).alias("sl"),
    )
    lens = lambda c: pl.col(c).list.len()
    inter = lambda a, b: pl.col(a).list.set_intersection(pl.col(b)).list.len()
    tri = lambda a, b: pl.when((pl.col(a) == "") | (pl.col(b) == "") | pl.col(a).is_null() | pl.col(b).is_null()).then(-1).when(pl.col(a) == pl.col(b)).then(1).otherwise(0).cast(pl.Int8)
    s2 = s.select(
        lens("qn").alias("qn_len"), lens("sn").alias("sn_len"), inter("qn", "sn").alias("n_inter"),
        lens("qa").alias("qa_len"), lens("sa").alias("sa_len"), inter("qa", "sa").alias("a_inter"),
        lens("qm").alias("qm_len"), lens("sm").alias("sm_len"), inter("qm", "sm").alias("m_inter"),
        tri("qm1", "sm1").alias("num1_eq"), tri("q_state", "s_state").alias("state_eq"), tri("q_pin", "s_pin").alias("pin_eq"),
        tri("qn1", "sn1").alias("first_tok_eq"),
        pl.when((lens("ql") == 0) | (lens("sl") == 0)).then(-1)
          .when(pl.col("ql").list.sort().list.join(" ") == pl.col("sl").list.sort().list.join(" ")).then(1)
          .when(inter("ql", "sl") > 0).then(2).otherwise(0).cast(pl.Int8).alias("legal_eq"),
    )
    s2 = s2.with_columns(
        (pl.col("n_inter") / (pl.col("qn_len") + pl.col("sn_len") - pl.col("n_inter")).clip(lower_bound=1)).alias("n_jac"),
        (pl.col("n_inter") / pl.min_horizontal("qn_len", "sn_len").clip(lower_bound=1)).alias("n_cont"),
        (pl.col("a_inter") / (pl.col("qa_len") + pl.col("sa_len") - pl.col("a_inter")).clip(lower_bound=1)).alias("a_jac"),
        (pl.col("a_inter") / pl.min_horizontal("qa_len", "sa_len").clip(lower_bound=1)).alias("a_cont"),
        (pl.col("m_inter") / pl.min_horizontal("qm_len", "sm_len").clip(lower_bound=1)).alias("m_cont"),
    )
    del s
    wn = _weighted_overlap(p.select("pid", "country", pl.col("q_ncore").alias("a"), pl.col("s_ncore").alias("b")), "a", "b", name_idf, "wn")
    wa = _weighted_overlap(p.select("pid", "country", pl.col("q_atoks").alias("a"), pl.col("s_atoks").alias("b")), "a", "b", addr_idf, "wa")
    wn = p.select("pid").join(wn, on="pid", how="left").drop("pid")
    wa = p.select("pid").join(wa, on="pid", how="left").drop("pid")
    meta = p.select(
        "rid", "s1", "bscore", "nkeys", "brank", "btop",
        (pl.col("bscore") / pl.col("btop")).alias("brel"),
        pl.col("q_is_domain").alias("is_domain"), pl.col("q_is_dba").alias("is_dba"), pl.col("q_is_native").alias("is_native"),
        (pl.col("q_atoks") == "").cast(pl.Int8).alias("q_addr_empty"),
        pl.col("q_ncore").str.len_chars().alias("qn_chars"), pl.col("s_ncore").str.len_chars().alias("sn_chars"),
    )
    out = pl.concat([meta, feats, s2, wn, wa], how="horizontal")
    # query-level context
    out = out.with_columns(
        pl.len().over("rid").alias("q_ncand"),
        pl.col("n_tset").rank("max", descending=True).over("rid").alias("n_tset_rank"),
        pl.col("a_tset").rank("max", descending=True).over("rid").alias("a_tset_rank"),
        (pl.col("wn_wj") + pl.col("wa_wj")).alias("w_sum"),
    )
    out = out.with_columns(
        pl.col("w_sum").rank("max", descending=True).over("rid").alias("w_sum_rank"),
        (pl.col("w_sum") - pl.col("w_sum").max().over("rid")).alias("w_sum_gap"),
    )
    return add_rank_feats(out)
