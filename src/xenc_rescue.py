"""Re-score the rescue candidates (records that blocking gave no candidate, rescue_*.py) with a cross-encoder.

The rescue model's top-5 candidates per record (xextra parquet parts, scored in the Colab notebook ->
xenc2_extra_scores.csv.gz with split rtrain / rtest) get a stacked score: logistic regression on
[logit p_rescue, cross-encoder logit, product], fitted on the train rescue records (2-fold by record for the check).

usage:
  python xenc_rescue.py <work> eval  <extra_scores.csv.gz>                 # local US+India check (needs v17_local_preds.parquet)
  python xenc_rescue.py <work> apply <extra_scores.csv.gz> <out.parquet>   # test: rid, s1, p_rescue (top-5 per record, re-scored)
then run rescue_apply.py with <out.parquet> instead of rescue_pred_test.parquet.
"""
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_eval import LocalEval, K_V10  # noqa: E402

W, STEP, XS = sys.argv[1], sys.argv[2], sys.argv[3]
TOPK = 5


def lg(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def feats(d):
    a = lg(d["p_rescue"].to_numpy())
    x = d["xr"].to_numpy()
    return np.column_stack([a, x, a * x])


def scores(split):
    return (pl.read_csv(XS, schema_overrides={"rid": pl.UInt32, "s1": pl.UInt32, "score": pl.Float64})
              .filter(pl.col("split") == split).select("rid", "s1", pl.col("score").alias("xr")).unique(["rid", "s1"]))


def top(df):
    return df.with_columns(pl.col("p_rescue").rank("ordinal", descending=True).over("rid").alias("r")).filter(pl.col("r") <= TOPK)


def train_rows():
    return top(pl.read_parquet(f"{W}/rescue_oof_train.parquet")).join(scores("rtrain"), on=["rid", "s1"])


def eval_():
    tr = train_rows()
    print("train rescue pairs with xenc", tr.height, "records", tr["rid"].n_unique(), "positives", int(tr["label"].sum()))
    fold = (tr["rid"].hash(seed=7) % 2).to_numpy()
    y = tr["label"].to_numpy()
    newp = np.empty(tr.height)
    X = feats(tr)
    for k in (0, 1):
        m = LogisticRegression(max_iter=2000).fit(X[fold != k], y[fold != k])
        newp[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    tr = tr.with_columns(pl.Series("p_new", newp))
    from sklearn.metrics import roc_auc_score
    print(f"AUC p_rescue {roc_auc_score(y, tr['p_rescue']):.4f}  xenc {roc_auc_score(y, tr['xr']):.4f}  stacked {roc_auc_score(y, newp):.4f}")
    t1_old = tr.sort("p_rescue", descending=True).group_by("rid").first()
    t1_new = tr.sort("p_new", descending=True).group_by("rid").first()
    print("top-1 correct: old", int(t1_old["label"].sum()), " new", int(t1_new["label"].sum()))
    le = LocalEval(W)
    pv = pl.read_parquet(os.environ.get("LOCAL_PREDS", f"{W}/v20_local_gbm.parquet"))
    cc = le.s1.select("s1", "country")
    base = []
    for c in ("US", "India"):
        k = K_V10[c] * float(os.environ.get("KSCALE", "1.75"))
        base.append(pv.join(cc.filter(pl.col("country") == c), on="s1")
                      .select("rid", "s1", (pl.col("p2x") * k / (pl.col("p2x") * k + 1 - pl.col("p2x"))).alias("_p")))
    base = pl.concat(base)

    def with_rescue(t1, col, odds):
        r = (t1.join(cc, on="s1").select("rid", "s1", (pl.col(col) / odds / (pl.col(col) / odds + 1 - pl.col(col))).alias("_p"))
               .filter(pl.col("_p") >= 0.05))
        return le.score(pl.concat([base, r]), col="_p", prior_adj=False, verbose=False)["US+India"]
    b0 = le.score(base, col="_p", prior_adj=False, verbose=False)["US+India"]
    print(f"no rescue {b0:.5f}")
    for odds in (1.0, 2.0, 3.0):
        o, n = with_rescue(t1_old, "p_rescue", odds), with_rescue(t1_new, "p_new", odds)
        print(f"odds/{odds}: old rescue {o:.5f} ({o - b0:+.5f})   new rescue {n:.5f} ({n - b0:+.5f})   diff {n - o:+.5f}")


def apply_(out):
    tr = train_rows()
    m = LogisticRegression(max_iter=2000).fit(feats(tr), tr["label"].to_numpy())
    print("coef", np.round(m.coef_[0], 3), "icpt", round(float(m.intercept_[0]), 3))
    allp = pl.read_parquet(f"{W}/rescue_pred_test.parquet")
    qc = pl.read_parquet(f"{W}/test_q.parquet", columns=["rid", "country"])
    allp = allp.join(qc, on="rid", how="left")
    countries = [c.strip() for c in os.environ.get("XENC_COUNTRIES", "US,India").split(",")]
    keep = allp.filter(~pl.col("country").is_in(countries)).select("rid", "s1", "p_rescue")   # France: unchanged (final France rows come from v13)
    te = top(allp.filter(pl.col("country").is_in(countries))).join(scores("rtest"), on=["rid", "s1"])
    te = te.with_columns(pl.Series("p_rescue", m.predict_proba(feats(te))[:, 1]))
    res = pl.concat([te.select("rid", "s1", "p_rescue"), keep])
    res.write_parquet(out)
    print("test rescue pairs re-scored", te.height, "records", te["rid"].n_unique(), "| kept unchanged", keep.height, "->", out)


if __name__ == "__main__":
    {"eval": lambda: eval_(), "apply": lambda: apply_(sys.argv[4])}[STEP]()
