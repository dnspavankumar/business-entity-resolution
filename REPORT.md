# Project Report: Business Entity Resolution

**Amazon ML Challenge 2026 · Team Ekdant.exe**
Gurram Ruthwik Goud · Nagasri Pavan Kumar Dharmapuri · Manne Lakshmi Varaprasad · Pattaswamy Vishwak Yashasree

| | |
|---|---|
| Final submission | **v21** |
| Best public leaderboard | **0.989191** (v20), up from 0.973 for our first submission |
| Candidate set | 3.98 candidates per Source-1 entity, with 97.7% of true pairs kept |
| Top of the leaderboard | about 0.9915 |

---

## 1. The problem

**Task.** Source-1 is a clean reference table of businesses (name and address). Source-2 and Source-3 hold noisy copies of those businesses. For every Source-1 entity, we must list the Source-2/3 records that are the same business.

**What makes it hard:**
- **Noise.** Records contain typos, dropped words, swapped legal forms (inc → corp), abbreviations, leet-speak, invented "pseudo-names", web-domain names ("tirupaticonsultingprivate.com") and names in native Indian scripts.
- **Decoys.** The test set contains near-copies of real businesses that belong to *nobody*. A decoy accepted as a match is a false positive.
- **Three countries, labels for two.** Train has US and India. Test adds **France**, which has no labels at all.
- **Scale.** Millions of records. Comparing every pair within a country would be about 6.7 × 10¹² comparisons.

**Metric.** Macro F0.5 per Source-1 entity. Precision counts more than recall. An entity with no true match scores 1 only if we predict nothing for it. So every false positive costs a lot.

**Rules.** No external data or APIs. Pretrained models only with an MIT/Apache licence and at most 8B parameters. Late in the competition the organisers added that the **candidate set is also judged**: blocking must scale, and fewer candidates per entity rank higher.

**Outputs.** `matching_results.tsv` (entity → matched ids) and `candidate_pairs.tsv` (entity → candidate ids). Every matched id must be inside the candidate set.

---

## 2. Solution overview

```
records ──► normalise ──► blocking (inverted index) ──► stage-1 ranker ──► stage-2 matcher ──► cross-encoders ──► selection ──► post-processing
            translit,      ~50 candidates/record         best entity        decoy-aware           (GPU, uncertain     expected F0.5     decoy filter,
            legal forms,   linear cost                   per record          LightGBM              pairs only)         per entity        France rules,
            numbers                                                                                                                      rescue search
```

1. **Normalise** names and addresses (transliteration of Indic scripts, accents, leet-speak, legal forms, street abbreviations, states, numbers, PIN codes).
2. **Block.** Composite keys over name and address tokens, scoped by country, looked up in an inverted index with frequency caps. This gives about 50 candidates per record.
3. **Stage-1 ranker** (LightGBM, 2-fold). Keeps each record's single best Source-1 entity.
4. **Stage-2 matcher** (LightGBM). A decoy-aware model scores that pair with about 64 similarity, house-number-offset, legal-form, decoy-word and cluster-context features.
5. **Cross-encoders** (DistilBERT and XLM-R, fine-tuned on a free Colab T4). They read both texts together and re-score only the uncertain pairs. A LightGBM stacker combines them with the stage-2 score.
6. **Selection.** Scores are shifted to each country's test prior. For each entity, we choose the set of records with the highest **expected F0.5**, and the empty set is one of the options.
7. **Post-processing.** A decoy post-filter, label-free France rules and a rescue search for records that blocking never found. France keeps its own validated decisions.

---

## 3. Challenges and how we tackled each one

### 3.1 Local validation lied (first submission: local 0.9855, leaderboard 0.973)
**Problem.** Our first model looked excellent on held-out train data but scored far lower on the leaderboard.
**Cause.** We measured the decoy density from candidate counts: the test has about **3× more decoys per entity** than train (US ×3.00, India ×3.04). Validation on train data therefore under-counted false positives.
**Solution.** We built a local mini-leaderboard (`src/local_eval.py`, `evaluation/evaluate.py`):
- It uses held-out subsets V (3%) and T (4%) of the train entities, never seen by the model that scores them (cross-fitting).
- Every wrong pair whose record matches nothing is weighted ×3.0 (US) or ×3.04 (India), so the decoy density matches the test.
- It applies the same prior shift and per-entity selection as the real submission.

**Result.** From then on, local and leaderboard agreed within about 0.0004. Local predicted +0.0015 for v15 and the leaderboard gave +0.0015. Uploads were limited, so this harness was what let us keep improving.

### 3.2 Decoys: model the adversary, not generic noise
**Problem.** Decoys look almost exactly like true matches.
**Investigation.** On train labels we found that decoys follow rules (the "decoy fingerprint"):
- The house number is the Source-1 number **plus an offset** from {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}. The offset is positive 99.8% of the time, and all decoys of one entity share one offset (97.8%).
- The name usually changes too: a legal-form swap, or an added word (Holdings, Group, Industries; in France Développement, "& Fils").
- Other kinds: the same address with the first word replaced, or the same name at another location.
- True-match typos, by contrast, are **symmetric**: −k is as common as +k.

**Solution.**
- Decoy-aware stage-2 features: signed house-number difference, offset membership, legal-form swap direction, self-supervised "decoy word" scores, cluster context.
- Negatives weighted ×2.6 in training, plus a per-country prior shift, to match the test density.
- A **post-filter** (`decoy_postfilter.py`) that removes pairs in an entity's positive-offset cells.
- An **upward-offset cap**: pairs shifted up by a decoy offset can never score above the older v10 model.

**Result.** Leaderboard 0.973 → 0.976 (post-filter) → 0.979 (decoy-aware matcher).

### 3.3 Global thresholds hurt precision
**Problem.** Lowering the threshold to gain recall (v3) made the score *worse*, because more decoys got in.
**Solution.** We replaced the single threshold with **expected-F0.5 selection per entity**. For each entity we consider the candidate sets ordered by score, compute the expected F0.5 of each from the calibrated probabilities, and keep the best. "Predict nothing" is one option, so entities without a true match are handled correctly.

### 3.4 France had no labels
**Problem.** France is about 15% of test entities, and we had nothing to train or validate on. A probe upload with France left empty showed France was far worse than US/India. Our first France rule (v11) was partly wrong.
**Solution.** Two **label-free checks** that work on the test itself:
- **Count profile** (`evaluation/prof_diff.py`). True noise grows with entity size, and entities with no match have almost none. Decoys are flat across sizes. From the size profile of a set of pairs, we can estimate what share of it is decoys.
- **Symmetry check** (`evaluation/sym_check.py`). True typos are equally +k and −k. Decoys are only +k. Any excess of accepted +k pairs means decoys.

With these we wrote France rules (`france_fix2.py`): reject business-type-word swaps ("Dunkerque Club EURL" → "Dunkerque Collège EURL") and accept harmless noise suffixes ("France", "Services").
**Result.** The count profile predicted +0.0050; the leaderboard gave **+0.0049** (0.980524 → 0.985429). It was our biggest single jump.
**Lesson.** Models trained on US/India do not transfer to France. Twice (the v15 matcher and the v18 cross-encoder for France) the label-free checks showed that a new model would add decoys or remove true French pairs. France therefore keeps its validated v13 decisions.

### 3.5 A local gain can be a test loss
**Problem.** "Sibling-agreement" features (another record of the entity has the same deviation) gave +0.0004 locally.
**Cause.** The symmetry check on the test showed more upward-offset decoys accepted. The test contains **decoy groups**: a decoy entity often has 2+ records at the same shifted number (19% of US strong-offset pairs vs 1.2% in train), so "another record agrees with me" is a trap on the test.
**Solution.** We dropped those features, added the upward-offset cap, and made the symmetry check mandatory before any upload.
**Follow-up lesson.** A score-driven change can also look asymmetric on the test for good reasons: true +1/+2 typos sit near the decision boundary because decoys use +1/+2 too. So we always compare the test pattern with the same analysis on labelled local data before calling it decoys.

### 3.6 Records that blocking never finds
**Problem.** Some records got no candidate at all: pseudo-names, glued web-domain names, native script, or common names with partial addresses.
**Solution.**
- Learned Indic → Latin transliteration dictionary (`learn_translit.py`, train labels only).
- A **rescue search** (`rescue_*.py`): a TF-IDF search for records with no candidate, a LightGBM model to score them, and a new selection for the affected entities (+0.00066 locally).
- In v21, the rescue's top-5 candidates are re-scored by a cross-encoder. Rescue AUC 0.9788 → 0.9933, and correct top-1 picks went from 16,194 to 17,134 (+0.0002 locally).

### 3.7 The LightGBM matcher hit a plateau
**Problem.** More data helped less and less. Training on 0.16M → 2.9M rows took the local score from 0.98684 to 0.98802, and a further doubling would give only about +0.0001.
**Solution.** A **cross-encoder** that reads both records' "name | address" together:
- `distilbert-base-multilingual-cased` (Apache-2.0, 135M parameters), fine-tuned on 646K train pairs, 2 epochs, about 35 minutes on a free Colab T4.
- Only the about 1.1M uncertain test pairs (0.002 < p2 < 0.998) are re-scored, so the GPU cost stays small.
- Alone it is weaker than the matcher (AUC 0.946 vs 0.979), but its errors are different. Stacked, the AUC reaches 0.988, and false positives fall from 325 to 190 (US).

**Result.** v17: local +0.0014, leaderboard **+0.0020** (0.988917).
- A second model of similar size (XLM-R base, v19) added only +0.00014: the two models mostly agree.
- A LightGBM stacker with pair decoy/noise features and a slightly larger prior shift (v20) gave +0.00012 locally and +0.00019 on the leaderboard (**0.989191**).

### 3.8 Candidate set must scale and stay small
**Problem.** The organisers announced that `candidate_pairs.tsv` and its code would be reviewed: no all-pairs comparison, and smaller is better.
**Solution.** Each record looks up its own keys in a hashed inverted index of Source-1, with per-key frequency caps, so the cost is linear in the number of records. The funnel:

| stage | pairs | per Source-1 entity |
|---|---|---|
| all pairs within a country (never computed) | ≈ 6.7 × 10¹² | ≈ 3.9M |
| key retrieval | 52.9M | 30.5 |
| learned ranker, best entity per record (+ rescue) | **6.89M** | **3.98** |
| final matches | 5.83M | 3.36 |

On labelled held-out data, the candidate set keeps 97.7% of true pairs (3.62 candidates per entity vs 3.47 true matches). We chose not to cut it further with the stage-2 score: that would be cosmetic and would make the candidates the same as the matches.

### 3.9 Limited hardware
**Problem.** The work machine had 2 CPUs, about 6 GB RAM and no GPU.
**Solution.** Everything is chunked: one country per process, Float32 casts, LightGBM matrices filled column group by column group from parquet. Peak RAM stays under about 6 GB for a 5-hour CPU run. All GPU work runs in free Colab notebooks (`gpu_crossencoder/`), and the CPU pipeline exports and imports parquet/CSV files.

### 3.10 The irreducible part: empty-address twins
**Problem.** About 60% of the remaining loss comes from records with **no address** whose name belongs to 2+ Source-1 entities.
**Investigation.** We tested every signal we could find: file order (50.2%), ID number (49.9%), ID mod-m leaks, address properties (AUC 0.53), entity size, Bayesian count priors, the Source-2/3 split, spelling copies, a pseudo-name character model (AUC 0.68), and sibling propagation of aliases. None of them beat chance.
**Conclusion.** These records are coin flips. We stopped spending time on them and focused on the fixable buckets.

---

## 4. Evaluation strategy

Uploads were limited, so every change went through these gates before upload:
1. `evaluation/evaluate.py`: a clear local gain on US+India (≥ about +0.0003; below about 0.0001 is noise).
2. `evaluation/sym_check.py`: no rise in accepted upward-offset pairs on the test.
3. `evaluation/prof_diff.py`: no decoy-like pairs added in France.
4. `evaluation/check_submission.py` plus the official validator: format PASS.

The last upload must always be a good file, because the private leaderboard decides the final ranking.

---

## 5. Results by version

| version | key change | public LB |
|---|---|---|
| v1 | blocking + LightGBM ranker + matcher, global threshold | 0.973 |
| v3 | lower threshold | 0.972 (worse) |
| v4–v5 | decoy house-number post-filter | 0.976549 |
| v8–v9 | decoy-aware stage-2 matcher | 0.97974 |
| v11 | decoy-word scores, prior shift, expected-F0.5 selection, first France rule | 0.980524 |
| v12 | France rules found with the count profile | 0.985429 |
| v13 | rescue search | (local +0.00066) |
| v15 | positional house-number features, 3.9× training rows, offset cap | 0.986916 |
| v16 | more training data (T4), prior shift ×1.5 | (local +0.0003) |
| v17 | + DistilBERT cross-encoder on uncertain pairs | 0.988917 |
| v18 | cross-encoder for France too | rejected by label-free checks |
| v19 | + XLM-R cross-encoder | 0.989000 |
| v20 | LightGBM stacker + prior shift ×1.75 | **0.989191** |
| **v21** | cross-encoder re-scoring of rescue candidates | final submission (local +0.0002) |

---

## 6. What did not work

- Larger LightGBM models, more trees, seed averaging, blending with older models.
- Isotonic recalibration, per-country prior values, "unreachable-aware" selection.
- Sibling-agreement features (they hurt on the test, §3.5).
- Every signal for the empty-address twins (§3.10).
- The cross-encoders for France (v18).
- Removing the upward-offset cap for the US (≈ 530 decoys added).
- Bigger stackers (they overfit).

Each was ≤ +0.0001 locally or unsafe on the test. Details are in `docs/APPROACHES_AND_LESSONS.md`.

---

## 7. Where the remaining loss is (estimated, v20)

| cause | ≈ LB loss | share | fixable? |
|---|---|---|---|
| Empty-address records with a shared name | 0.0064 | ≈ 60% | no signal found |
| Records blocking never finds | 0.0014 | ≈ 13% | partly |
| France errors apart from the twins | 0.0012 | ≈ 11% | partly |
| True matches rejected or ranked second | 0.0010 | ≈ 9% | partly |
| Decoys accepted | 0.0009 | ≈ 8% | partly |

Estimated per country: US/India ≈ 0.990, France ≈ 0.985. France has 1.7× more coin-flip records than US/India.

---

## 8. Future work (ranked)

1. **Re-rank every record's top-5 candidates** with the cross-encoders, not only the best pair (+0.0003–0.0006 locally).
2. **Dense retrieval** (multilingual-e5 / LaBSE) for records that blocking misses (+0.0002–0.0004).
3. **A bigger cross-encoder** (xlm-roberta-large or bge-reranker-v2-m3) trained on many more pairs (+0.0002–0.0005).
4. **A France model from pseudo-labels**, checked with the count profile (+0.0002–0.0005 on the LB).
5. An address-first matcher that tolerates number typos, for pseudo-names.
6. Splitting web-domain names into words before blocking.

Items 1–4 together might reach about 0.990, at the cost of several days of GPU work.

---

## 9. Key lessons

1. **Make validation match the test's decoy density**, or it will mislead you.
2. **Study the adversary.** The decoy fingerprint gave most of the gains.
3. **Choose per entity by expected F0.5**, not with a global threshold.
4. **Label-free checks** (count profile, offset symmetry) can validate a country with no labels.
5. **Check local gains against test-side checks.** Train and test can differ in subtle ways (decoy groups).
6. **A different kind of model** (a cross-encoder) beats more of the same model.
7. **Know what is irreducible** and stop spending time on it.
