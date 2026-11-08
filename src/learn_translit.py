"""Learn an Indic-script token -> Latin token dictionary from TRAIN ground-truth pairs.

For every true (Source1, Source2/3) pair where the S2/S3 name is written in an Indic
script and has the same number of whitespace tokens as the (Latin) Source 1 name, the
tokens are aligned positionally and co-occurrence counts are accumulated.  The most
frequent Latin token becomes the translation.  Same for state names in addresses.
"""
import collections, json, os, re, sys
import polars as pl

IND = r"[ऀ-ൿ]"
TOK_RE = re.compile(r"[ऀ-ൿ‌‍]+")


def lat_tokens(s):
    s = s.lower().replace("&", " and ").replace("+", " and ").replace("'", "").replace(".", "")
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def nat_tokens(s):
    out = []
    for t in s.split():
        m = TOK_RE.findall(t)
        k = "".join(m).replace("‌", "").replace("‍", "")
        out.append(k if k else None)
    return out


def main(data_dir, out_path):
    gt = (pl.read_parquet(f"{data_dir}/train_ground_truth.parquet")
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
          .filter(pl.col("m") != ""))
    s1 = pl.read_parquet(f"{data_dir}/train_source1.parquet").select(
        pl.col("entity_id").alias("source1_entity_id"), pl.col("business_name").alias("n1"),
        pl.col("business_address").alias("a1"))
    o = pl.concat([pl.read_parquet(f"{data_dir}/train_source{i}.parquet") for i in (2, 3)]).filter(
        pl.col("business_name").str.contains(IND) | pl.col("business_address").str.contains(IND)).select(
        pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    j = gt.join(o, on="m").join(s1, on="source1_entity_id")
    cnt = collections.defaultdict(collections.Counter)
    for n1, n2 in zip(j["n1"].to_list(), j["n2"].to_list()):
        if not re.search(IND, n2):
            continue
        a, b = lat_tokens(n1), nat_tokens(n2)
        if len(a) == len(b):
            for x, y in zip(b, a):
                if x:
                    cnt[x][y] += 1
    # addresses: native phrase vs the Latin state component of the Source 1 address
    import normalize as N
    states = set(N.IN_STATES.keys())
    for a1, a2 in zip(j["a1"].to_list(), j["a2"].to_list()):
        if not re.search(IND, a2):
            continue
        st = [c.strip().lower() for c in a1.split(",") if c.strip().lower() in states and len(c.strip()) > 3]
        if not st:
            continue
        for comp in a2.split(","):
            if not re.search(IND, comp):
                continue
            nt = [x for x in nat_tokens(comp) if x]
            lt = st[0].split()
            if len(nt) == 1:
                cnt[nt[0]][st[0]] += 1
            elif len(nt) == len(lt):
                for x, y in zip(nt, lt):
                    cnt[x][y] += 1
    d = {}
    for k, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= 2:
            d[k] = y
    json.dump(d, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("dictionary size", len(d))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
