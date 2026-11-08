"""Normalise all source files -> parquet with normalised columns."""
import sys, time
from multiprocessing import Pool
import polars as pl
import normalize as N


def _proc(args):
    names, addrs, countries = args
    out = {k: [] for k in ["nfull", "ncore", "nalt", "is_domain", "is_dba", "atoks", "anums", "state", "pin", "acity"]}
    for n, a, c in zip(names, addrs, countries):
        full, core, alt, dom, dba = N.normalize_name(n)
        toks, nums, st, pin, cc = N.normalize_address(a, c)
        out["nfull"].append(" ".join(full)); out["ncore"].append(" ".join(core)); out["nalt"].append(" ".join(alt))
        out["is_domain"].append(dom); out["is_dba"].append(dba)
        out["atoks"].append(" ".join(toks)); out["anums"].append(" ".join(nums)); out["state"].append(st)
        out["pin"].append(pin); out["acity"].append("|".join(cc))
    return out


def prep_file(inp, outp, nproc=2, chunk=50000):
    df = pl.read_parquet(inp)
    names = df["business_name"].fill_null("").to_list(); addrs = df["business_address"].fill_null("").to_list()
    cs = df["country"].fill_null("").to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], cs[i:i + chunk]) for i in range(0, len(names), chunk)]
    del names, addrs, cs
    parts = []
    with Pool(nproc) as p:
        for r in p.imap(_proc, jobs):
            parts.append(pl.DataFrame(r, schema={k: (pl.Int8 if k in ("is_domain", "is_dba") else pl.Utf8) for k in r}))
    del jobs
    out = pl.concat([df, pl.concat(parts)], how="horizontal")
    out = out.with_columns(pl.col("business_name").str.contains(r"[ऀ-ൿ]").cast(pl.Int8).alias("is_native"))
    out.write_parquet(outp)
    return out.height


if __name__ == "__main__":
    data, work = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] if len(sys.argv) > 3 else [f"{s}_source{i}" for s in ("train", "test") for i in (1, 2, 3)]
    for nm in names:
        t = time.time()
        n = prep_file(f"{data}/{nm}.parquet", f"{work}/{nm}_norm.parquet")
        print(nm, n, f"{time.time()-t:.0f}s", flush=True)
