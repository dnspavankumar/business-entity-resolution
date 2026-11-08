"""Blocking-key generation (vectorised with polars).

All keys are scoped by country and hashed to u64.  Key kinds:
  0 NA : core-name token  x  address token          (name + address agree)
  1 NN : pair of core-name tokens                   (name only, word-order free)
  2 AA : address number x address word / word pair  (address only - catches alias/DBA names)
  3 F  : full sorted core name
  4 T  : single core-name token (or 6-char prefix of concatenated name)
Each kind has its own document-frequency cap (see CAPS); keys whose Source-1
frequency exceeds the cap are dropped as non-selective.
"""
import polars as pl

ADDR_GENERIC = ["st", "rd", "ave", "dr", "blvd", "ln", "ct", "pl", "sq", "hwy", "pkwy", "cir", "ter", "trl",
                "ste", "apt", "fl", "bldg", "unit", "po", "box", "rue", "near", "opposite", "main", "cross",
                "ground", "and", "des", "the", "city", "way", "floor", "nagar", "road", "sector", "phase",
                "complex", "colony", "indl", "estate", "dist", "vill", "apts", "chemin", "allee", "impasse",
                "lane", "street", "off", "wing", "block", "office", "tower", "plaza", "pvt", "ltd", "1st",
                "2nd", "3rd", "4th", "5th"]
MAX_NT = 4
MAX_NUM = 3
MAX_ALPHA = 9
CAPS = {0: 50, 1: 50, 2: 30, 3: 50, 4: 20}
# caps used for query records that have no usable address (name-only retrieval)
CAPS_NOADDR = {0: 50, 1: 300, 2: 30, 3: 300, 4: 300}
S1_MAXCAP = 300

EMPTY = pl.lit([], dtype=pl.List(pl.Utf8))


def token_lists(df: pl.DataFrame) -> pl.DataFrame:
    cc = pl.col("ncore").str.replace_all(" ", "")
    nt = (pl.concat_str([pl.col("ncore"), pl.col("nalt")], separator=" ").str.split(" ")
          .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique(maintain_order=True).list.head(MAX_NT))
    pref = pl.when(cc.str.len_chars() >= 6).then(pl.concat_list([pl.lit("~") + cc.str.slice(0, 6)])).otherwise(EMPTY)
    longp = (pl.col("ncore").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 8))
             .list.eval(pl.lit("~") + pl.element().str.slice(0, 6)))
    pref = pl.concat_list([pref, longp]).list.unique(maintain_order=True)
    toks = (pl.col("atoks").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(ADDR_GENERIC) & (pl.element() != "")))
            .list.unique(maintain_order=True))
    nums = toks.list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.head(MAX_NUM)
    alph = (toks.list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$") & (pl.element().str.len_chars() >= 3)))
            .list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True, maintain_order=True)).list.head(MAX_ALPHA))
    pin = pl.when(pl.col("pin") != "").then(pl.concat_list([pl.lit("pin") + pl.col("pin")])).otherwise(EMPTY)
    return df.select("rid", "country", nt.alias("nt"), pref.alias("pref"), nums.alias("nums"), alph.alias("alph"), pin.alias("pin"),
                     pl.col("ncore").str.split(" ").list.sort().list.join(" ").alias("fk"))


def _h(col, country, seed):
    return (pl.col("country") + ":" + pl.col(col)).hash(seed=seed)


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Returns (rid:u32, key:u64, kind:i8)."""
    tl = token_lists(df)
    n = tl.select("rid", "country", pl.concat_list(["nt", "pref"]).alias("t")).explode("t").drop_nulls("t").select("rid", _h("t", None, 1).alias("th"), pl.col("t"))
    n_only = n.filter(~pl.col("t").str.starts_with("~")).select("rid", "th", "t")
    a = (tl.select("rid", "country", pl.concat_list(["nums", "alph", "pin"]).alias("a")).explode("a").drop_nulls("a")
         .select("rid", _h("a", None, 2).alias("ah")))
    na = n.join(a, on="rid").select("rid", (pl.col("th") ^ (pl.col("ah") // 2)).alias("key"), pl.lit(0, pl.Int8).alias("kind"))
    nn = (n_only.join(n_only, on="rid", suffix="2").filter(pl.col("t") < pl.col("t2"))
          .select("rid", (pl.col("th") ^ (pl.col("th2") // 4) ^ 0x9E3779B97F4A7C15).alias("key"), pl.lit(1, pl.Int8).alias("kind")))
    num = tl.select("rid", "country", "nums").explode("nums").drop_nulls("nums").select("rid", _h("nums", None, 3).alias("h1"))
    al = tl.select("rid", "country", "alph").explode("alph").drop_nulls("alph").select("rid", _h("alph", None, 4).alias("h2"), pl.col("alph"))
    aa1 = num.join(al.select("rid", "h2"), on="rid").select("rid", (pl.col("h1") ^ (pl.col("h2") // 8)).alias("key"))
    al4 = al.group_by("rid", maintain_order=True).head(4)
    aa2 = (al4.join(al4, on="rid", suffix="b").filter(pl.col("alph") < pl.col("alphb"))
           .select("rid", (pl.col("h2") ^ (pl.col("h2b") // 16) ^ 0x7F4A7C159E3779B9).alias("key")))
    aa = pl.concat([aa1, aa2]).with_columns(pl.lit(2, pl.Int8).alias("kind"))
    f = tl.filter(pl.col("fk") != "").select("rid", (pl.col("country") + "|F|" + pl.col("fk")).hash(seed=5).alias("key"), pl.lit(3, pl.Int8).alias("kind"))
    t = n.select("rid", pl.col("th").alias("key"), pl.lit(4, pl.Int8).alias("kind"))
    out = pl.concat([na, nn, aa, f, t]).unique(["rid", "key"])
    return out.with_columns(pl.col("rid").cast(pl.UInt32))


def make_keys_chunked(df: pl.DataFrame, chunk: int = 200000) -> pl.DataFrame:
    return pl.concat([make_keys(df.slice(i, chunk)) for i in range(0, df.height, chunk)])
