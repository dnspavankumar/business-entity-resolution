"""Local mini-leaderboard: scores stage-2 probabilities for US + India the way the leaderboard scores the test.

What it does (see ../src/local_eval.py for the engine):
  * held-out Source-1 subsets V (3%) + T (4%) of train, about 154K entities, with their true matches;
  * macro F0.5 over ALL of them (singletons included: an empty prediction scores 1 only for a singleton);
  * TEST decoy density: a wrong pair whose record matches nothing is counted x3.00 (US) / x3.04 (India),
    because the test has about 3x more decoys per entity than train;
  * the same per-country prior shift (odds x K) and expected-F0.5 selection as the submission;
  * US and India weighted by their test sizes.
It agreed with the leaderboard to about 0.0004 (0.9871 local vs 0.9867 derived from a France-empty probe).

usage:
  python evaluate.py <work_dir> <preds.parquet> [p2_column] [--kscale 1.5]
  preds = rid, s1, <p2> for the V+T best pairs, e.g. work/cf_preds_v14_T2.parquet (written by
          `stage2_v14.py train`: cross-fitted, so V and T are never scored by a model trained on them).
Reference values: v10 matcher 0.98690, v15 0.98787, v16 (+T4) 0.98802.
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from local_eval import LocalEval, K_V10  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ks = 1.0
if "--kscale" in sys.argv:
    ks = float(sys.argv[sys.argv.index("--kscale") + 1])
    args = [a for a in args if a != str(sys.argv[sys.argv.index("--kscale") + 1])]
W, path = args[0], args[1]
col = args[2] if len(args) > 2 else "p2"
k = {c: K_V10[c] * ks for c in K_V10}
print(f"K = {k}")
LocalEval(W).score(pl.read_parquet(path), col, k=k)
