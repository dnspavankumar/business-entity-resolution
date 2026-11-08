"""Local "mini leaderboard" for US + India (the countries that have train labels).

It scores a set of stage-2 probabilities on the held-out Source-1 subsets V (3%) and T (4%) the same way the
leaderboard scores the test set, but with the TEST decoy density:
  * the test has more decoy candidates per Source-1 entity than train (US x3.0, India x1.94, measured from the
    test candidate counts), so every accepted wrong pair is counted ratio_c times (a fractional copy by hash);
  * probabilities are shifted to the test prior per country exactly like PRIOR_ADJ in stage2_v2.predict
    (odds x 2.6 / ratio_c; the stage-2 model itself is trained with negatives weighted x2.6);
  * the expected-F0.5 selection (ev_select) picks the matches per entity;
  * macro F0.5 over ALL V/T entities, including singletons and true matches that blocking never found.
The US and India scores are combined with the test weights (US 663,106 / India 809,986 entities).

France has no labels, so it cannot be scored here.

usage (module):  from local_eval import LocalEval ; le = LocalEval(work_dir) ; le.score(pred_df)  # pred_df: rid, s1, p2
usage (CLI):     python local_eval.py <work_dir> <preds.parquet> [p2_column]
"""
import os
import sys

import numpy as np
import polars as pl

# decoy density in the test relative to train (other-entity errors are NOT more frequent in the test):
# (test candidates per S1 - positives - other-entity negatives) / decoy negatives, per country, from V u T
RATIO = {"US": 3.0014, "India": 3.0430}
# stage-2 odds correction used by PRIOR_ADJ in v10/v11 (from a pooled pos/neg estimate)
K_V10 = {"US": 0.8648, "India": 1.3400}
TEST_S1 = {"US": 663106, "India": 809986}
TRAIN_RATIO = 2.6


def _ftable(n, cache={}):
    if n in cache:
        return cache[n]
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
    cache[n] = (z, F)
    return z, F


def ev_select(best, col="p2", max_n=12, fallback_thr=0.8):
    b = best.select("rid", "s1", col).sort(["s1", col], descending=[False, True])
    g = b.group_by("s1", maintain_order=True).agg(pl.col("rid"), pl.col(col).alias("p"), pl.len().alias("n"))
    rows, frames = [], []
    for n in sorted(g["n"].unique().to_list()):
        gg = g.filter(pl.col("n") == n)
        if n > max_n:
            frames.append(gg.explode(["rid", "p"]).filter(pl.col("p") >= fallback_thr).select("rid", "s1"))
            continue
        z, F = _ftable(n)
        P = np.clip(np.array(gg["p"].to_list(), dtype=np.float64), 1e-6, 1 - 1e-6)
        prob = np.exp(np.log(P) @ z.T + np.log(1 - P) @ (1 - z).T)
        k = (prob @ F).argmax(1)
        rids, s1s = gg["rid"].to_list(), gg["s1"].to_list()
        for i in np.nonzero(k)[0]:
            for r in rids[i][: k[i]]:
                rows.append((r, s1s[i]))
    out = [pl.DataFrame(rows, schema={"rid": best.schema["rid"], "s1": best.schema["s1"]}, orient="row")] if rows else []
    return pl.concat(out + frames)


class LocalEval:
    def __init__(self, W, all_neg_dup=False):
        self.W = W
        self.all_neg_dup = all_neg_dup
        s1all = pl.concat([pl.read_parquet(f"{W}/val_s1.parquet"), pl.read_parquet(f"{W}/T_s1.parquet")]).unique()
        cc = pl.read_parquet(f"{W}/train_s1.parquet", columns=["rid", "country"]).rename({"rid": "s1"})
        self.s1 = s1all.join(cc, on="s1")
        gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").rename({"true_s1": "s1"})
        self.gt = gt.join(self.s1.select("s1"), on="s1").select("s1", "rid")
        self.ntrue = self.gt.group_by("s1").len().rename({"len": "n_true"})
        # records that are a true match of SOME Source-1 entity: a wrong pair with such a record is an
        # "other-S1" error (ambiguity), whose density does not change in the test; only pairs with records
        # that match nothing (decoys / unrelated) are more frequent in the test.
        self.matched_rids = gt.select("rid").unique().with_columns(pl.lit(True).alias("is_match_rec"))

    def _macro(self, pred, s1c):
        """pred: s1, rid, w (weight of the pair: 1 for true, ratio for wrong) ; returns mean F0.5 over s1c."""
        lab = pred.join(self.gt.with_columns(pl.lit(1).alias("t")), on=["s1", "rid"], how="left").with_columns(pl.col("t").fill_null(0))
        per = lab.group_by("s1").agg((pl.col("t") * pl.col("w")).sum().alias("tp"), pl.col("w").sum().alias("np"))
        x = s1c.select("s1").join(per, on="s1", how="left").join(self.ntrue, on="s1", how="left").fill_null(0)
        tp, npred, nt = x["tp"].to_numpy(), x["np"].to_numpy(), x["n_true"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(npred > 0, tp / npred, 0.0)
            R = np.where(nt > 0, tp / nt, 0.0)
            f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
        f = np.where((nt == 0) & (npred == 0), 1.0, f)
        return float(f.mean())

    def score(self, preds, col="p2", prior_adj=True, select=None, verbose=True, k=None):
        """preds: rid, s1, <col> for the V/T best pairs. Returns dict with per-country and combined scores."""
        res = {}
        for c in ("US", "India"):
            s1c = self.s1.filter(pl.col("country") == c)
            p = preds.join(s1c.select("s1"), on="s1")
            if prior_adj:
                kc = (k or K_V10)[c]
                p = p.with_columns((pl.col(col) * kc / (pl.col(col) * kc + 1 - pl.col(col))).alias("_p"))
            else:
                p = p.with_columns(pl.col(col).alias("_p"))
            acc = select(p) if select is not None else ev_select(p, "_p")
            lab = (acc.join(self.gt.with_columns(pl.lit(1).alias("t")), on=["s1", "rid"], how="left").with_columns(pl.col("t").fill_null(0))
                      .join(self.matched_rids, on="rid", how="left").with_columns(pl.col("is_match_rec").fill_null(False)))
            # test-like: a wrong pair whose record matches nothing (decoy) counts ratio_c times (expected value);
            # a wrong pair whose record belongs to another entity counts once (old behaviour: all_neg_dup=True)
            if self.all_neg_dup:
                wexpr = pl.when(pl.col("t") == 1).then(1.0).otherwise(RATIO[c])
            else:
                wexpr = pl.when(pl.col("t") == 1).then(1.0).when(pl.col("is_match_rec")).then(1.0).otherwise(RATIO[c])
            pred_tl = lab.with_columns(wexpr.alias("w")).select("s1", "rid", "w")
            pred_tr = lab.with_columns(pl.lit(1.0).alias("w")).select("s1", "rid", "w")
            res[c] = dict(test_like=self._macro(pred_tl, s1c), train_like=self._macro(pred_tr, s1c),
                          n_acc=acc.height, fp=int((lab["t"] == 0).sum()), n_s1=s1c.height)
        wt = {c: TEST_S1[c] / sum(TEST_S1.values()) for c in TEST_S1}
        res["US+India"] = sum(wt[c] * res[c]["test_like"] for c in wt)
        if verbose:
            print(f"LOCAL  US {res['US']['test_like']:.5f}  India {res['India']['test_like']:.5f}  ->  US+India {res['US+India']:.5f}"
                  f"   (train-like US {res['US']['train_like']:.5f} India {res['India']['train_like']:.5f}; FP {res['US']['fp']}/{res['India']['fp']})", flush=True)
        return res


if __name__ == "__main__":
    W = sys.argv[1]
    col = sys.argv[3] if len(sys.argv) > 3 else "p2"
    LocalEval(W).score(pl.read_parquet(sys.argv[2]), col)


# ---------------------------------------------------------------------------------------------
# expected-F0.5 selection that knows some true matches of an entity may be OUTSIDE the candidates
# (blocking misses, empty-address records assigned to a twin entity). With probability u(n) the
# entity has one extra true match that we cannot predict; this lowers the value of "predict nothing".
_FTAB_U = {}


def _ftable_u(n):
    if n in _FTAB_U:
        return _FTAB_U[n]
    z = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    T = z.sum(1)
    F0 = np.zeros((2 ** n, n + 1)); F1 = np.zeros((2 ** n, n + 1))
    F0[:, 0] = (T == 0).astype(float)          # nothing predicted, no extra true record
    F1[:, 0] = 0.0                              # nothing predicted, but one true record exists outside
    for k in range(1, n + 1):
        tp = z[:, :k].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = tp / k
            for F, extra in ((F0, 0), (F1, 1)):
                R = np.where(T + extra > 0, tp / (T + extra), 0.0)
                F[:, k] = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    _FTAB_U[n] = (z, F0, F1)
    return _FTAB_U[n]


U_OF_N = {0: 0.028, 1: 0.105, 2: 0.090, 3: 0.077, 4: 0.070, 5: 0.067, 6: 0.055, 7: 0.053}


def ev_select_u(best, col="p2", max_n=12, fallback_thr=0.8, u_of_n=None, u_scale=1.0):
    u_of_n = u_of_n or U_OF_N
    b = best.select("rid", "s1", col).sort(["s1", col], descending=[False, True])
    g = b.group_by("s1", maintain_order=True).agg(pl.col("rid"), pl.col(col).alias("p"), pl.len().alias("n"))
    rows, frames = [], []
    for n in sorted(g["n"].unique().to_list()):
        gg = g.filter(pl.col("n") == n)
        if n > max_n:
            frames.append(gg.explode(["rid", "p"]).filter(pl.col("p") >= fallback_thr).select("rid", "s1"))
            continue
        z, F0, F1 = _ftable_u(n)
        u = min(1.0, u_scale * u_of_n.get(n, 0.053))
        F = (1 - u) * F0 + u * F1
        P = np.clip(np.array(gg["p"].to_list(), dtype=np.float64), 1e-6, 1 - 1e-6)
        prob = np.exp(np.log(P) @ z.T + np.log(1 - P) @ (1 - z).T)
        k = (prob @ F).argmax(1)
        rids, s1s = gg["rid"].to_list(), gg["s1"].to_list()
        for i in np.nonzero(k)[0]:
            for r in rids[i][: k[i]]:
                rows.append((r, s1s[i]))
    out = [pl.DataFrame(rows, schema={"rid": best.schema["rid"], "s1": best.schema["s1"]}, orient="row")] if rows else []
    return pl.concat(out + frames)
