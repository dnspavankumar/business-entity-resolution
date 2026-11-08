# GPU cross-encoders (used by v17, v19, v20 and the final v21)

**Idea.** LightGBM sees hand-made similarity features. A small transformer that reads both texts together, "name | address" of the record and of the Source-1 entity, can catch patterns the features miss: pseudo-names, partial addresses, word swaps. Its score is **stacked** with the LightGBM score p2, and only for uncertain pairs (0.002 < p2 < 0.998). Confident pairs are left unchanged.

- **Model:** `distilbert-base-multilingual-cased` (Apache-2.0, 135M parameters, allowed: ≤ 8B, open licence). It is fine-tuned only on the provided train data.
- **Hardware:** free Google Colab T4 GPU, about 1 hour. The cloud box and the user's PC VM have no GPU.

## Notebooks (free Colab T4; packages in `requirements-gpu.txt`)
| notebook | model | trains on | scores | output | used by |
|---|---|---|---|---|---|
| `xenc_colab.ipynb` | distilbert-base-multilingual-cased (Apache-2.0, 135M) | xtrain (T2–T4 pairs) | xvt + xtest (uncertain band) | `xenc_scores.csv.gz` | v17 → v21 |
| `xenc2_xlmr_colab.ipynb` | xlm-roberta-base (MIT, 278M) | xtrain | xvt + xtest (+ optional xextra) | `xenc2_scores.csv.gz` | v19 → v21 |
| `xenc3_rescue_colab.ipynb` | distilbert-base-multilingual-cased | xtrain | xextra (rescue top-5, `src/xenc_export_rescue.py`) | `xenc3_extra_scores.csv.gz` | v21 |

- **Stacking:**
  - v17/v19 used a logistic-regression stacker (removed; superseded by the v20 LightGBM stacker).
  - v20/v21: `src/xenc_stack_gbm.py` (LightGBM + pair features, K×1.75).
  - v21 rescue: `src/xenc_rescue.py`.

## Steps
1. **Export pairs** (in the pipeline work dir): `python src/xenc_export.py <work> <out_dir>`
   - `xtrain_part*.parquet`: 646,749 labelled pairs from the training subsets T2–T4 (hard pairs with 0.02 ≤ p1 ≤ 0.999, plus 60K easy ones).
   - `xvt_part*.parquet`: 56,516 uncertain V+T pairs with labels, for the local check and for fitting the stacker.
   - `xtest_part*.parquet`: 1,100,575 uncertain test pairs.
2. **Colab:**
   1. Upload `xenc_colab.ipynb` and set Runtime → T4 GPU.
   2. Upload the 12 parquet files and Run all.
   3. The notebook trains for 2 epochs (batch 128, lr 5e-5, fp16, max length 128) and prints the hold-out AUC.
   4. It downloads `xenc_scores.csv.gz` with columns `rid, s1, score (logit), split ∈ {vt, test}`.
3. **Local check:** `python src/xenc_stack_gbm.py <work> eval xenc_scores.csv.gz,xenc2_scores.csv.gz`
   - It fits a LightGBM stacker on `[logit p2, both cross-encoder logits, logit p1, country, pair decoy/noise features]` over the V+T band pairs (2-fold by Source-1, negatives ×2.6 so the scale matches p2).
   - It prints the local leaderboard with and without the cross-encoders.
4. **Apply** (only if the gain is clear): `bash run.sh stack <dataset> <work> <out> xenc_scores.csv.gz xenc2_scores.csv.gz`
   - It keeps the v15 upward-offset cap and changes only US and India by default (`XENC_COUNTRIES`).
   - This also runs the downstream steps (decoy post-filter → `france_fix2.py` → `rescue_apply.py` → `merge_countries.py`, France from v13) and exports the rescue candidates for `xenc3_rescue_colab.ipynb`.

**Result (v17):**
- Band AUC: p2 0.9789, cross-encoder alone 0.9462, stacked 0.9878.
- Local US+India: 0.98816 → **0.98955** (+0.0014) at K×1.5.
- False positives: US 325 → 190, India 325 → 171.
- Test (label-free): about 1,900 fewer accepted +k offset pairs, with −k unchanged, so the removed pairs are decoys.
- The Colab run took about 35 minutes on a T4. `run.sh` automates everything except the Colab steps.
