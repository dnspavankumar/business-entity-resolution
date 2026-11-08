# Evaluation engine

We can't upload often, so every change is tested here first. There are two kinds of check.

## 1. Local mini-leaderboard (US + India, with labels): `evaluate.py`
The engine is `../src/local_eval.py` (class `LocalEval`, function `ev_select`).

**How it scores:**
- It uses the held-out train subsets V (3%) + T (4%), about 154K Source-1 entities, which are never used to train the matcher that scores them (cross-fit).
- It computes macro F0.5 over all these entities, singletons included, like the leaderboard.
- **It copies the test decoy density.** The test has about 3× more decoys than train, so every wrong pair whose record matches nothing counts ×3.00 (US) or ×3.04 (India). Without this, local scores are about 0.01 too optimistic (the v1 mistake).
- It uses the same prior shift (odds × K, `K_V10`) and the same expected-F0.5 selection as the submission.
- It weights US and India by their test sizes.

```bash
python evaluate.py <work_dir> <work_dir>/cf_preds_v14_T2.parquet            # -> LOCAL US .. India .. -> US+India 0.98787 (v15)
python evaluate.py <work_dir> <work_dir>/cf_preds_v14_T2.parquet p2 --kscale 1.5
```
- The input is `rid, s1, p2` for the V+T best pairs. `stage2_v14.py train` writes it (cross-fitted).
- Reference values: v10 0.98690, v15 0.98787, v16 0.98802.
- The LB agreed: local +0.0015 for v15 became +0.0015 on the LB.
- Differences below about 0.0001 are noise.

## 2. Label-free checks on the test (all countries, including France)
**`prof_diff.py <work> old.tsv new.tsv`** (count profile)
- For the pairs the new file ADDS or REMOVES, it estimates the share that are decoys.
- It relies on the entity-size profile: true noise ≈ [0.28, 1, 0.82, 0.72, 0.65, 0.60, 0.53], decoys flat.
- It predicted the v12 France gain exactly.
- It is **biased** when the change depends on model scores or cluster context. Use it for rule-based changes.

**`sym_check.py <work> file1.tsv [file2.tsv]`** (symmetry)
- It counts accepted pairs with house-number offset +k vs −k.
- True noise is symmetric; decoys are only +k.
- Compare two files: if `excess_strong` goes up, the new file accepts more decoys.
- The absolute value is not zero, because number parsing is not perfectly symmetric (India) and the post-filter removes +k pairs.
- This check caught the sibling-feature problem (see `../docs/APPROACHES_AND_LESSONS.md`, L5).

## 3. Format check: `check_submission.py <output_dir> <dataset>/test`
It checks, with little memory:
- one row per Source-1 entity;
- no duplicate ids;
- ids start with S2-/S3-;
- every matched id is inside `candidate_pairs.tsv`.

It prints PASS or FAIL. Also run the official `student_resource/utils/validate_submission.py` (flag `--candidate`) on a machine with enough RAM.

## Rule for uploading
Upload only if:
- `evaluate.py` shows a clear gain (≥ about +0.0003);
- `sym_check.py` shows no rise in upward-offset pairs;
- `prof_diff.py` shows no decoy-like additions in France;
- the format check passes.
