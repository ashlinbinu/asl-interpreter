# Evaluation

Generated 2026-10-28 by `python -m scripts.evaluation_report` on Windows 11, AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD, Python 3.12.10. All accuracy numbers are on signers **not seen in training**.

## Words (50-sign vocabulary, ASL Citizen)

Model `words-20261027-175106` (bigru), feature spec 1.0. Protocol: **official_signer_split** (train 32 / val 5 / test 11 signers; clips 737 / 184 / 633). Verified (≥3 held-out test signers).

| Metric (held-out test signers) | Value |
|---|---|
| Top-1 accuracy | **84.0%** |
| Top-5 accuracy | 95.3% |
| Macro-F1 | 0.840 |
| ECE after temperature scaling | 0.031 |
| Per-signer accuracy (min / mean / max) | 67.9% / 83.7% / 96.3% |
| Reject threshold (95% precision on val) | 0.56 (val coverage 87.0%) |

Candidates (chosen by validation macro-F1; test shown for transparency):

| Model | Val macro-F1 | Test top-1 | Epochs | Fit time |
|---|---|---|---|---|
| cnn | 0.876 | 82.0% | 79 | 86.4 s |
| bigru | 0.878 | 84.0% | 57 | 190.7 s |

Weakest signs (test F1): thank you 0.46, good 0.50, happy 0.52, stop 0.57, help 0.62, mother 0.64, what 0.70, school 0.70.

Most frequent confusions (test): good → thank you 33%, happy → please 25%, like → happy 25%, thank you → good 25%, where → deaf 23%, yes → bathroom 20%, come → happy 18%, again → come 17%.

### Practice score

logistic(target probability + DTW component percentiles), fitted on val signers. Ranking correct sign vs. a different sign on test signers: **AUC 0.993** (classifier probability alone 0.993; 2532 pairs). Single components: handshape 0.80, orientation 0.79, location 0.75, movement 0.63, two_hand 0.69. The components add little ranking power beyond the classifier; they are shown to explain the score and drive the hints.

### Scoring against your own recording (custom signs, self-recorded letters)

Signs with recordings but no trained model are scored by DTW against their own clips. Calibrated on the same person making a different sign (`scripts.calibrate_own_clip`), fitted on train + val signers (combination `all_components`, chosen by signer-grouped CV AUC: all_components 0.827, mean_of_components 0.795).

Proxy test on held-out ASL Citizen signers, same signer repeating a word vs the same signer making a different word: **AUC 0.841** (91 repeat / 657 different-sign pairs); median score 93.0 vs 30.5. Single components: handshape 0.84, orientation 0.87, location 0.78, movement 0.67, two_hand 0.71. **Unverified on your own custom signs**: no held-out data exists for them.

## Alphabet (24 static letters)

Model `alphabet-20261027-175212` (mlp). Sources: chicago_fsvid, kaggle_asl_alphabet; supplementary (train-only): kaggle_asl_alphabet. Test signers: chicago_fsvid:rita (domain: hand_crop).

**UNVERIFIED**: verification needs ≥3 held-out *full-frame video* signers (has 0). The test signer is a ChicagoFSVid studio hand crop recoloured for MediaPipe, so this measures handshape recognition across signers in that domain, **not webcam accuracy**.

| Metric | Value |
|---|---|
| Test accuracy (signer_holdout) | 75.0% |
| Test macro-F1 | 0.659 |
| Top-5 | 96.0% |
| ECE | 0.036 |
| Leave-one-signer-out (mlp, no augmentation) | mean 77.1% (andy 79.5%, drucie 84.9%, rita 75.4%, robin 68.5%) |
| Without the Kaggle photos (same test signer) | 76.8% |

Most frequent confusions: D → K 51%, O → N 30%, I → T 27%, W → V 20%, E → N 18%, E → O 18%, U → R 17%, I → Y 14%.

Confusable sets: AEMNST 71.4%, UVR 77.0%, KP 81.8%, GQ 87.5%, GH 74.4%.

## Latency (measured on this machine)

DTW runs on **numba**.

| Step | Time |
|---|---|
| Features for one clip (FEATURE_SPEC, Python; ~58 frames) | 21.5 ms |
| Word classification (numpy) | 3.8 ms |
| Word score incl. DTW vs 5 references (numpy; DTW on numba) | 33.7 ms |
| Word classification in JavaScript (Node, same code as the browser) | 10.8 ms |

MediaPipe landmark extraction offline: ~60 ms/frame per process on this CPU (hand ≈33, pose ≈22); throughput 38 frames/s with 3 processes (PLAN.md P2).
Browser FPS and end-to-end live latency were **not measured**: no browser with a webcam was available in this environment. The camera shows its FPS live.

## Data

Generated 2026-10-28 by `python -m scripts.dataset_report --write-docs`. Signers are counted per source (`source:signer`).

| Category | Source | Signs | Clips | Signers | Clips by split |
|---|---|---|---|---|---|
| alphabet | chicago_fsvid | 24 | 3780 | 4 | none 3780 |
| alphabet | kaggle_asl_alphabet | 24 | 7374 | 1 | none 7374 |
| user | user | 4 | 4 | 2 | none 4 |
| word | asl_citizen | 50 | 1554 | 48 | test 633, train 737, val 184 |

Words with ≥15 clips and ≥8 signers: **50 of 50**.
Letters with clips: 24 of 26; fewest signers for any letter: 0.

## Limitations

- A learning aid, not a translator. Text → sign is word-by-word glossing; ASL grammar, word order and non-manual markers (facial grammar) are not modelled.
- Words: 50 signs, one variant each (the ASL Citizen gloss with most clips; e.g. EAT1, DRINK1). Other regional or variant forms will score low. Accuracy is for ASL Citizen's webcam recordings; your camera, lighting and framing can lower it.
- Alphabet: unverified (see above). J and Z are not recognised or scored: no multi-signer motion data.
- Custom signs and letters you record yourself are scored against your own recordings only (unverified; see the proxy test above). A sign with no recording can't be scored.
- The practice score's weights were fitted on 5 validation signers; treat it as an estimate.
- Several datasets are non-commercial or research-only (docs/DATASET.md). Don't deploy publicly or commercially with them.
