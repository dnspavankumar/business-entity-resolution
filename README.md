# Business Entity Resolution: Amazon ML Challenge 2026 (team Ekdant.exe)

This project matches noisy business records (Source-2 and Source-3) to a reference table (Source-1) in the US, India and France. The test set also contains deliberate **decoys**: near-copies of a business that belong to nobody.

- **Final submission: v21.** The metric is macro F0.5 per Source-1 entity.
- **Best public leaderboard score:** 0.989191 (v20). v21 adds +0.0002 locally on top of it.
- **Full write-up:** [`REPORT.md`](REPORT.md) covers the problem, every challenge we hit and how we solved it, the results and future work.

## Folder layout
```
README.md                  this file: layout and how to run
REPORT.md                  complete project report
run.sh                     the single pipeline script (steps: stage1 | cpu | stack | final | check)
requirements.txt           CPU packages (Python 3.11)
src/                       pipeline code (table below)
models/                    trained models behind the submitted file
evaluation/                local mini-leaderboard + label-free test checks + format check (see evaluation/README.md)
  deep/                    loss breakdown, miss kinds, twin rates, offset cells, leak checks
gpu_crossencoder/          Colab notebooks for the 3 cross-encoder runs + requirements-gpu.txt
docs/
  APPROACHES_AND_LESSONS.md  detailed log: every version, lesson and dropped idea
  DEEP_ANALYSIS.md           detailed error analysis and future-work list
```

## Run (final v21)
```bash
pip install -r requirements.txt
DS=/path/to/student_resource/dataset

# (optional, first time) retrain the stage-1 ranker; models/ already holds trained copies
# bash run.sh stage1 $DS work out_v1   # then copy work/model_stage1_* and work/model_stage2_cf.* into models/

# 1. CPU: data -> blocking -> features -> matchers -> export of the uncertain pairs   (about 5 h on 2 cores, < 6 GB RAM)
bash run.sh cpu $DS work output
# 2. GPU (free Colab T4): gpu_crossencoder/xenc_colab.ipynb and xenc2_xlmr_colab.ipynb on work/colab
#    -> xenc_scores.csv.gz, xenc2_scores.csv.gz
# 3. stack the cross-encoders (v20, also written to output/ as a fallback) and export the rescue candidates
bash run.sh stack $DS work output xenc_scores.csv.gz xenc2_scores.csv.gz
# 4. GPU: gpu_crossencoder/xenc3_rescue_colab.ipynb (xtrain parts from work/colab + work/colab3) -> xenc3_extra_scores.csv.gz
# 5. final v21 and the format check
bash run.sh final $DS work output xenc3_extra_scores.csv.gz
bash run.sh check $DS work output
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir $DS/test
```
Each GPU run takes about 35–70 minutes on a T4.

## Code map (`src/`)
| step | scripts | what it does |
|---|---|---|
| data | `convert.py` | TSV → parquet |
| normalise | `prep.py`, `normalize.py`, `learn_translit.py` | Indic transliteration (`resources/indic_token_dict.json`), accents, leet-speak, legal forms, street abbreviations, states, numbers, PIN |
| blocking | `run_blocking.py`, `keys.py`, `blocking.py` | composite-key inverted index, per-kind frequency caps, top 50 by summed IDF |
| features | `build_features.py`, `features.py`, `select_T.py`, `select_T2.py`, `prep_T2.py` | about 64 pairwise features; held-out subsets V, T and training subsets T2–T4 |
| stage 1 | `crossfit.py`, `train.py`, `metrics.py`, `make_submission.py` | 2-fold LightGBM ranker (best entity per record) |
| stage 2 | `stage2.py`, `stage2_v2.py`, `decoy_features.py`, `stage2_v14.py`, `extra_feats.py`, `extra_test.py` | decoy-aware matcher, prior shift, expected-F0.5 selection, upward-offset cap |
| post-processing | `decoy_postfilter.py`, `france_fix.py`, `france_fix2.py`, `merge_countries.py` | offset-cell decoy filter, France rules, US/India + France merge |
| rescue | `rescue_queries.py`, `rescue_cands.py`, `rescue_score.py`, `rescue_apply.py` | TF-IDF search for records blocking missed |
| cross-encoders | `xenc_export.py`, `xenc_stack_gbm.py`, `xenc_export_rescue.py`, `xenc_rescue.py` | GPU export, v20 stacker, v21 rescue re-scoring |
| evaluation | `local_eval.py` | local leaderboard engine used by `evaluation/` |

## Models (`models/`)
- `model_stage1_f0/f1.txt` + `model_stage1_cf.json`: stage-1 ranker (2 folds). `model_stage2_cf.*`: base feature list.
- `s2v2_cf_final.txt`: v10 matcher (France). `s2v2_cf_final_dw.txt`: decoy-word model.
- `s2v2_cf_v16.txt` + `s2v2_v16.json`: v16 matcher (US/India).
- `xenc_stack_gbm_v20.txt`: v20 LightGBM stacker. `xenc_rescue_v21.json`: v21 rescue stacker. `model_rescue.txt`: rescue model.
- The run rebuilds all stage-2 and rescue models in `work/`; these copies document the submitted file.

**Licences:** LightGBM (MIT); `distilbert-base-multilingual-cased` (Apache-2.0, 135M) and `xlm-roberta-base` (MIT, 278M), fine-tuned only on the provided train data. No external data or APIs.
