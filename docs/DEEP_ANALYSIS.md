# Deep analysis: why we are at 0.989 and what could reach 0.99+

Written 2026-09-27 (deadline day), after v19. Every number below can be reproduced with the scripts in `evaluation/deep/` and `evaluation/`.

| | score |
|---|---|
| **Best leaderboard (v19)** | **0.989000** |
| Local US+India (before the rescue) | 0.98969 |
| Top teams | about 0.9915 |
| Needed for 0.99 | +0.001 on the leaderboard |

---


## 0. Where the whole remaining loss goes (leaderboard estimate, v20 = 0.989191)
The total loss is 1 − 0.989191 = **0.0108**. It splits by country weight (US+India 85% of entities, France 15%) and by the local bucket shares.

| cause | ≈ LB loss | share | fixable? |
|---|---|---|---|
| Empty-address records whose name belongs to 2+ entities (US/India 0.0053, France ≈ 0.0011) | **0.0064** | **≈ 60%** | ❌ coin flip, no signal in the data |
| Records blocking never finds (fake names, glued web names, native script, common names with a partial address) | 0.0014 | ≈ 13% | partly: dense retrieval, better transliteration |
| France errors apart from the twins (no labels to train or check on) | 0.0012 | ≈ 11% | partly: France pseudo-labels |
| True matches rejected or ranked second | 0.0010 | ≈ 9% | partly: re-ranking, bigger models |
| Wrong matches accepted (decoys, look-alike names) | 0.0009 | ≈ 8% | partly: bigger models |

Estimated per country: US/India ≈ 0.990, France ≈ 0.985.
- Gap to 0.99 is 0.0008.
- Gap to the top (≈ 0.9915) is 0.0023, which is about all of the "partly fixable" rows together.

## 1. Where the remaining errors are (local, labelled US+India, v17)
`python evaluation/deep/loss_breakdown.py work work/v17_local_preds.parquet`: the gain if a whole bucket were fixed.

| bucket | pairs | oracle gain | fixable? |
|---|---|---|---|
| **Empty-address records** (all buckets) | 11,280 | **≈ 0.0063 (≈ 60%)** | ❌ When the name is shared by 2+ entities, the owner is a coin flip (section 5) |
| No candidate at all (with address) | 1,955 | 0.0016 | partly; the rescue already recovers ≈ 0.0007 |
| True S1 not among the candidates (with address) | 721 | 0.0007 | partly (section 2) |
| True S1 not ranked first (with address) | 792 | 0.0005 | partly (re-ranking) |
| Reachable but rejected (with address) | 1,148 | 0.0007 | partly (stronger text models) |
| All false positives | 361 | 0.0010 | partly (stronger text models) |

Even a perfect fix of every "with address" bucket gives about +0.004 locally. Realistic methods recover 10–30% of it.

## 2. What the misses with an address look like
`python evaluation/deep/miss_kinds.py work 5` (3,468 missed true pairs):

| kind | pairs | example (record → true Source-1) | why it is missed |
|---|---|---|---|
| pseudo-name, S1 address unique | 1,024 | "Lumrizasol, 681 WANDERING WAY, INDINAPOLIS" → "JKG Hyperliquid, LLC, 6814 Wandering Way, Indianapolis" | invented name; the address has typos or a dropped digit |
| name shared by 2+ S1 | 921 | "Balaji Services, ###11, Bangalore" → "Balaji Services Private Limited, No. 11, 4Th Cross …" | common Indian name and partial address, so it is ambiguous |
| web-domain name | 549 | "tirupaticonsultingprivate.com" → "Tirupati Consulting Private Limited" | the words are glued together |
| native script | 515 | "ভারত ইঞ্জিনিয়ারিং প্রাইভেট লিমিটেড, 1, KOLKATA" → "Bharat Engineering Private Limited" | the transliteration dictionary misses words |
| pseudo-name, S1 address shared | 310 | "Zephzeph, 1694 200 Road, Phillipsburg" (2 S1 at that address) | coin flip between co-located businesses |
| other | 149 | | |

## 3. Structural facts per country
`python evaluation/deep/twin_rates.py work`: empty-address records whose exact normalised name belongs to 2+ Source-1 entities, per 1,000 S1.

| | train US | train India | test US | test India | **test France** |
|---|---|---|---|---|---|
| empty-address "twin" records per 1,000 S1 | 24.3 | 20.6 | 18.2 | 20.1 | **33.2** |

- France has about **1.7× more coin-flip records** than US/India.
- That explains roughly half of France's lower score (France ≈ 0.984 vs US/India ≈ 0.990, estimated from the leaderboard). The rest is from having no France labels.

## 4. Test-side checks (no labels needed)
- **India, house number +1/+2** (`evaluation/deep/offset_cells.py`): the test accepts about 2,750 more +1/+2 pairs than the local true-pair reference predicts, so they are probably decoys.
  - They are spread over all cells, with decoy shares of 0.1–0.3. Removing a pair only pays above a share of about 0.3.
  - So there is nothing safe to remove. US shows no excess.
- **France, negative offsets:** France accepts only 60% of its −1/−2 pairs (US 97%), but that is just 338 pairs, a negligible gain.
- **Outside the uncertain band** (p2 > 0.998): accepted +k and −k pairs are balanced, so there is no need to widen the cross-encoder band.

## 5. Leak checks
`python evaluation/deep/leak_checks.py work`: **no leak.**
- S1 id mod m equals record id mod m at exactly the chance rate (m = 2…1000). The correlation of the ids is 0.0001.
- Records per entity: Source-2 0–5 and Source-3 0–6, a broad spread with no hard cap to exploit.
- Earlier sessions also found row order, ID order and twin order random.

## 6. Research (web + literature, 2026-09-27)
- There are no public write-ups of top solutions for this challenge. Public GitHub repos are generic blocking + LightGBM, scoring 0.62–0.98.
- **Literature ideas and their size here:**
  - calibration before the F-beta set selection (Lipton et al. 2014; Ye et al. 2012): about +0.0001–0.0003
  - an entity-level "has any match" model for the empty set (Dembczyński et al., NIPS 2011): uncertain
  - seed averaging: about +0.0001
  - comparing all of a record's candidates at once (ComEM, 2024)
  - larger rerankers (bge-reranker-v2-m3, Apache-2.0)
  - Ditto-style marker tokens for numbers and legal forms
- **A second cross-encoder of the same size gives little** (measured: XLM-R + DistilBERT = +0.00014).
- **Most likely reading of the top scores:** much stronger text models on the resolvable pairs, not a trick for the coin-flip records.

---

## 7. What could still move us to 0.99+ (for future work)

| # | idea | targets | expected local gain | effort |
|---|---|---|---|---|
| 1 | **Re-rank the top-5 candidates of every record with the cross-encoders.** Today only the best stage-1 pair is scored. The new best pair then goes through stage 2. | "not ranked first" / "not among candidates", pseudo-names at unique addresses | +0.0003–0.0006 | 1–2 days: export, about 3 h on a T4, re-run stage 2 |
| 2 | **Dense retrieval for records that blocking misses.** Multilingual sentence embeddings (multilingual-e5, MIT; LaBSE, Apache-2.0) with a k-nearest-neighbour search inside the state, fed into the rescue. | native script, web domains, pseudo-names | +0.0002–0.0004 | 1 day + GPU |
| 3 | **A bigger cross-encoder trained on far more pairs:** all train subsets plus all candidate negatives, 2–3 epochs, xlm-roberta-large or bge-reranker-v2-m3, on an A100. | false positives, rejected true pairs | +0.0002–0.0005 | 1 day + paid GPU |
| 4 | **A France-specific model from pseudo-labels:** confident v13 France matches as positives, business-type swaps and offset decoys as negatives. Checked with the count profile. | France (15% of entities, gap ≈ 0.006) | +0.0002–0.0005 on the LB | 1 day, risky without labels |
| 5 | **An address-first matcher for pseudo-names**, tolerant to number typos (digit dropped or added, ±1) | 1,334 pseudo-name misses | +0.0001–0.0003 | half a day |
| 6 | **Split web-domain names into words** using the Source-1 vocabulary before blocking | 549 domain misses | +0.0001 | 2 h |
| 7 | **Calibration** / entity-level empty-set model | selection errors | +0.0001–0.0003 | half a day |

**Checked on the evening of 09-27 after v20: all were noise (within ±0.00005 locally), so none were used.**
- **Stacker variants:** added positional number features; 3-seed bagging; a bigger LightGBM (31 leaves, 800 rounds) overfits: −0.00014.
- **Prior shift:** per-country K (US best ×2.5, India ×1.5–1.75: +0.00003).
- **Rescue odds divisor:** 1.0–3.0, all within ±0.00002.
- **Empty-address pairs:** separate odds multipliers for unique core name / unique full name / twins.
  - The model is already calibrated for the decision. Twins get worse with more odds (−0.0003 at ×3).
- **Empty-address "not ranked first" misses** (5,809):
  - 3,790 have a true S1 and a chosen S1 with IDENTICAL full names, a pure coin flip.
  - About 1,070 share the core name.
  - 857 differ only by a word that the "drop a word + add a generic suffix" noise explains either way, e.g. "Supreme Diagnostic" → "…Dynamics" or "…Services".
  - Only a generative noise model (likelihood of each edit) could separate those, which is multi-day work.
- **Test-side offsets:**
  - US accepts about 760 fewer strong +k pairs than the local true-pair reference. That is the price of the cap and the post-filter, worth about +0.00003 if fixed.
  - France is similar: about 575 pairs, about +0.00002.

**More checks, 09-27 21:10 IST (after v20 = 0.989191):**
- **France with the cross-encoder stack, split by address status:**
  - Pairs WITH an address: it removes 10,100 pairs (81% true by profile) and adds 4,598 (72% true). That is clearly worse.
  - Keeping only its additions is about neutral (+/−0.00008, sign uncertain). France stays on v13.
- **Removing the house-number cap for the US:**
  - Accepted +k pairs rise from 1,704 to 2,991 while −k stays at 3,079.
  - The true-pair reference (0.80 × −k ≈ 2,463) implies about 760 true pairs and about 530 decoys added, which is a net loss.
  - **So the cap is right and stays.**
- **Rescue top-1 recalibration** with gap features: AUC 0.962 → 0.966 but +0.00000 locally.

**Last checks, 09-27 22:20 IST:**
- **Pseudo-name aliases are shared inside an entity.** 32% of entities with 2+ pseudo-name records use the IDENTICAL invented name, and 43.5% share the first 4 letters, versus 0.005% / 0.1% for random pairs.
  - So we tested "sibling propagation": give an unmatched record the entity of an accepted record with the same name.
  - Local precision is 0.28 at best for empty-address records. With the pseudo-name constraint it is about 0: the unmatched copies of an accepted alias are **decoys**.
  - The true copies are already captured by the cluster-context features. No gain.
- **Third web search** (generator, Unstop forum, ≥0.990 claims):
  - No generator code is public.
  - No public write-up is above about 0.98 (repos: AvinashMalladi ">0.98", ViveKumar007 0.977, Akash-bardia 0.976, ayushmanjha52 0.971, harshgitty58 0.969).
  - ayushmanjha52 also lists the empty-address shared-name records as unsolved.

**Not worth more time:**
- the empty-address coin flips (no signal exists);
- LightGBM size, seeds or more rows (+0.0001 per doubling);
- a LightGBM stacker for the cross-encoders (+0.00008);
- sibling-agreement features (they hurt on the test).

**Honest estimate:** items 1–4 together might add about +0.001 on the leaderboard, which would reach about 0.990. That is several days of GPU work, not hours.
