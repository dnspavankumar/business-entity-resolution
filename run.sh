#!/usr/bin/env bash
# Business Entity Resolution (Amazon ML Challenge 2026): one entry point for the final v21 pipeline.
#
# usage: bash run.sh <step> <dataset_dir> [work_dir=work] [output_dir=output] [extra args]
#
#   stage1   <ds> [work] [out]                    (optional, first time) train the stage-1 ranker + transliteration
#                                                 dictionary from scratch; then copy work/model_stage1_* and
#                                                 work/model_stage2_cf.* into models/ (models/ already has trained copies)
#   cpu      <ds> [work] [out]                    data -> blocking -> features -> v13 (France) + v16 (US/India) matchers
#                                                 -> exports the uncertain pairs to <work>/colab for the GPU   (~5 h, 2 cores)
#            GPU: gpu_crossencoder/xenc_colab.ipynb + xenc2_xlmr_colab.ipynb on <work>/colab -> xenc_scores, xenc2_scores
#   stack    <ds> [work] [out] <xenc.csv> <xenc2.csv>
#                                                 v20: LightGBM stacker of p2 + both cross-encoders, post-processing,
#                                                 then exports the rescue candidates to <work>/colab3
#            GPU: gpu_crossencoder/xenc3_rescue_colab.ipynb (xtrain parts from <work>/colab + <work>/colab3) -> xenc3_extra_scores
#   final    <ds> [work] [out] <xenc3_extra.csv>  v21: cross-encoder re-scoring of the rescue candidates -> <out>/*.tsv
#   check    <ds> [work] [out]                    low-memory format check of <out> against <ds>/test
set -euo pipefail

STEP=${1:?step: stage1 | cpu | stack | final | check}
DATASET=$(realpath "${2:?dataset dir}"); WORK=$(realpath -m "${3:-work}"); OUT=$(realpath -m "${4:-output}")
HERE=$(cd "$(dirname "$0")" && pwd); MD=$HERE/models
mkdir -p "$WORK/data" "$OUT"
cd "$HERE/src"
GT="$WORK/data/train_ground_truth.parquet"

# decoy post-filter -> France rules -> rescue, for one scored version.   args: <tag> <rescue_pred> <BEST parquet>
postprocess() {
  local T=$1 RP=$2 B=$3
  mkdir -p "$WORK/${T}_ff" "$WORK/${T}_final"
  python3 decoy_postfilter.py --matching "$WORK/out_$T/matching_results.tsv" --dataset "$DATASET" \
          --out "$WORK/out_$T/matching_results_pf.tsv" --removed "$WORK/out_$T/removed_pf.tsv" --refit --hidden
  python3 france_fix2.py "$WORK" "$WORK/out_$T/matching_results_pf.tsv" "$WORK/out_$T/removed_pf.tsv" "$WORK/${T}_ff"
  BEST=$B python3 rescue_apply.py "$WORK" "$WORK/${T}_ff/matching_results.tsv" "$WORK/out_$T/removed_pf.tsv" \
          "$WORK/${T}_ff/france_fix2_pairs.parquet" "$RP" "$WORK/${T}_final" \
          "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_$T/candidate_pairs.tsv"
}
# final files: US + India from <tag>, France from v13
publish() {
  python3 merge_countries.py "$WORK" "$WORK/out_v13/matching_results.tsv" "$WORK/$1_final/matching_results.tsv" \
          "$OUT/matching_results.tsv" US,India
  cp "$WORK/$1_final/candidate_pairs.tsv" "$OUT/candidate_pairs.tsv"
  echo "done: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
}

case "$STEP" in
stage1)
  python3 convert.py "$DATASET" "$WORK/data"
  python3 learn_translit.py "$WORK/data" resources/indic_token_dict.json   # Indic -> Latin token dictionary (train GT only)
  python3 prep.py "$WORK/data" "$WORK"
  python3 run_blocking.py train "$WORK"
  python3 run_blocking.py test "$WORK"
  GT=$GT python3 build_features.py train "$WORK"        # V (3% of S1) + 2M random training queries
  python3 select_T.py "$WORK"
  GT=$GT python3 build_features.py trainT "$WORK"
  GT=$GT python3 build_features.py trainRest "$WORK"
  python3 build_features.py test "$WORK"
  python3 crossfit.py "$WORK" all                       # 2-fold stage-1 ranker + baseline stage-2 matcher
  python3 make_submission.py "$WORK" "$OUT"             # v1 baseline submission
  ;;

cpu)
  # ---- data, blocking, features
  python3 convert.py "$DATASET" "$WORK/data"
  python3 prep.py "$WORK/data" "$WORK"
  python3 run_blocking.py train "$WORK"
  python3 run_blocking.py test "$WORK"
  N_TRAIN_Q=1 GT=$GT python3 build_features.py train "$WORK"   # V (3% of S1)
  python3 select_T.py "$WORK"                                 # T (4% of S1)
  GT=$GT python3 build_features.py trainT "$WORK"
  python3 build_features.py test "$WORK"
  # ---- v13: v10 decoy-aware matcher + post-filter + France rules + rescue (France rows of the final file)
  mkdir -p "$WORK/out_v10" "$WORK/out_v12" "$WORK/out_v13"
  python3 stage2_v2.py "$WORK" "$MD" prep_train
  NEG_RATIO=2.6 python3 stage2_v2.py "$WORK" "$MD" train_cf
  S2KEY=cf_final SELECT=ev PRIOR_ADJ=1 python3 stage2_v2.py "$WORK" "$MD" predict "$WORK/out_v10"
  cp "$WORK/test_best_s2v2_cf_final.parquet" "$WORK/test_best_s2v2_cf_final_v10.parquet"
  python3 decoy_postfilter.py --matching "$WORK/out_v10/matching_results.tsv" --dataset "$DATASET" \
          --out "$WORK/out_v10/matching_results_v10.tsv" --removed "$WORK/out_v10/removed_v10.tsv" --refit --hidden
  python3 france_fix2.py "$WORK" "$WORK/out_v10/matching_results_v10.tsv" "$WORK/out_v10/removed_v10.tsv" "$WORK/out_v12"
  python3 rescue_queries.py "$WORK"
  python3 rescue_cands.py "$WORK" train "$WORK/rescue_q_train.parquet" "$WORK/rescue_cand_train.parquet"
  python3 rescue_cands.py "$WORK" test "$WORK/rescue_q_test.parquet" "$WORK/rescue_cand_test.parquet"
  python3 rescue_score.py "$WORK" train
  python3 rescue_score.py "$WORK" predict
  python3 rescue_apply.py "$WORK" "$WORK/out_v12/matching_results.tsv" "$WORK/out_v10/removed_v10.tsv" \
          "$WORK/out_v12/france_fix2_pairs.parquet" "$WORK/rescue_pred_test.parquet" "$WORK/out_v13" \
          "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v10/candidate_pairs.tsv"
  # ---- v16 matcher: extra training subsets T2, T3, T4 + positional house-number features, K x1.5
  for S in "T2 7 17" "T3 17 27" "T4 27 37"; do
    set -- $S
    python3 select_T2.py "$WORK" "$1" "$2" "$3"
    SUB=$1 python3 build_features.py trainT2 "$WORK"
    python3 prep_T2.py "$WORK" "$MD" "$1"
  done
  python3 stage2_v14.py "$WORK" "$MD" train                    # prints the local cross-fit check
  for C in US India France; do python3 extra_test.py "$WORK" "$C"; done
  mkdir -p "$WORK/out_v14"
  KSCALE=1.5 python3 stage2_v14.py "$WORK" "$MD" predict "$WORK/out_v14"
  # ---- export for the GPU cross-encoders
  mkdir -p "$WORK/v16"
  cp "$WORK/cf_preds_v14_T2.parquet" "$WORK/s2v2_v14.json" "$WORK/test_best_s2v2_cf_final.parquet" "$WORK/v16/"
  V15_DIR="$WORK/v16" python3 xenc_export.py "$WORK" "$WORK/colab"
  echo "cpu step done: run gpu_crossencoder/xenc_colab.ipynb and xenc2_xlmr_colab.ipynb on $WORK/colab, then: bash run.sh stack ..."
  ;;

stack)
  X1=$(realpath "${5:?xenc_scores.csv}"); X2=$(realpath "${6:?xenc2_scores.csv}")
  mkdir -p "$WORK/out_v20"
  V15_DIR="$WORK/v16" python3 xenc_stack_gbm.py "$WORK" eval "$X1,$X2"      # local check (prints the gain)
  V15_DIR="$WORK/v16" KSCALE=1.75 python3 xenc_stack_gbm.py "$WORK" apply "$X1,$X2" "$WORK/out_v20"
  postprocess v20 "$WORK/rescue_pred_test.parquet" "$WORK/out_v20/test_best_xenc.parquet"
  publish v20                                                               # v20 (LB 0.989191) as a fallback
  python3 xenc_export_rescue.py "$WORK" "$WORK/colab3"
  echo "stack step done: run gpu_crossencoder/xenc3_rescue_colab.ipynb on $WORK/colab3 (+ xtrain parts), then: bash run.sh final ..."
  ;;

final)
  X3=$(realpath "${5:?xenc3_extra_scores.csv}")
  LOCAL_PREDS="$WORK/v20_local_gbm.parquet" python3 xenc_rescue.py "$WORK" eval "$X3" || true   # local check
  python3 xenc_rescue.py "$WORK" apply "$X3" "$WORK/rescue_pred_test_x.parquet"
  mkdir -p "$WORK/v21_final"
  # v21 reuses the v20 post-filter and France outputs; only the rescue input changes
  BEST="$WORK/out_v20/test_best_xenc.parquet" python3 rescue_apply.py "$WORK" "$WORK/v20_ff/matching_results.tsv" \
          "$WORK/out_v20/removed_pf.tsv" "$WORK/v20_ff/france_fix2_pairs.parquet" "$WORK/rescue_pred_test_x.parquet" \
          "$WORK/v21_final" "$WORK/fr_lowp1_noise_pairs.parquet" "$WORK/out_v20/candidate_pairs.tsv"
  publish v21
  ;;

check)
  python3 "$HERE/evaluation/check_submission.py" "$OUT" "$DATASET/test"
  ;;

*) echo "unknown step: $STEP (use stage1 | cpu | stack | final | check)"; exit 1 ;;
esac
