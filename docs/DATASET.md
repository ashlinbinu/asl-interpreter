# Datasets

Every clip becomes one NPZ of raw landmarks (`clipstore.py`) plus a `reference_clips` row with `source`, `signer_id`,
`split`, `license` and `attribution`. Splits are always made **by signer**. Large data lives in
`HANDSPEAK_STORE_DIR` (outside OneDrive; D: has ~145 GB free on this machine).

## Sources

Sizes are from HTTP `HEAD` requests on 2026-10-27 (no content downloaded). "License" is what the source states; where
it states nothing, we treat the data as research-only and never redistribute it.

| Source id | What | Size / access | License | Status |
|---|---|---|---|---|
| `own` | Recorder page / `record_cli.py`, webcam video + landmarks | local | own recordings, signer consent | ready (P1) |
| `user` | "Add a sign" custom signs | local | user-contributed | ready |
| `asl_citizen` | Microsoft ASL Citizen: 83,399 isolated-sign videos, 2,731 ASL-LEX glosses, 52 signers, official signer splits (train 35 / val 6 / test 11 signers). **Inspected 2026-10-27: no letter glosses (0 of 26); all 50 words present** (29–39 clips, 26–31 signers each; *goodbye* is glossed `BYE`) | **45.9 GB** zip, direct link, HTTP ranges supported: only the 1,557 vocabulary clips (791 MB) are fetched | Microsoft research license: **non-commercial, non-revenue research only** | **ingested (words)** |
| `chicago_fsvid` | ChicagoFSVid (Kim et al. 2017), signers Andy, Drucie, Rita, Robin. **The public release is not video:** MATLAB 7.3 (HDF5) arrays of 128×128 single-channel **hand crops** for every frame of the fingerspelled words, a letter label per frame, 10 folds, plus HOG features. The isolated letters described in the paper are not in it. | **1.7 GB** `.tgz` (downloaded to `<store>/downloads/chicago_fsvid/`) | **none stated** ("released publicly"; cite Kim et al. 2017); research-only, kept local | **ingested (static letters)**: the centre of each letter run, recoloured for MediaPipe (skin tint, light background, ×2); hand found in Andy 1,152 / Drucie 1,185 / Rita 1,205 / Robin 238 frames. `domain=hand_crop`, so it never makes the alphabet "verified". Crops follow the hand, so letter location and Z's trajectory are lost. |
| `chicago_fswild` | ChicagoFSWild / FSWild+: 7,304 / 55,232 fingerspelled sequences, 160 / 260 signers, JPEG frames from online video | **14.3 GB** / **87.4 GB**, direct links | none stated; source videos belong to their creators | not planned: sequence labels only (no letter timing) |
| `kaggle_asl_alphabet` | Kaggle `grassknoted/asl-alphabet` by Akash: 3,000 200×200 photos per letter, **1 signer**, 24 meaningful static letters (J/Z skipped) | 1.1 GB, downloaded by you to `<store>/raw/asl_alphabet_train/` | **GPL 2** (Kaggle metadata) | **supplementary static training only, never validation/test**; 400 per letter tried, 7,374 with a detected hand (65–93% per letter). One photo per static letter is shown as the letter's reference photo (`scripts.build_reference_images`, attribution and GPL 2 shown under it) |
| `fsboard` | FSboard (Google, CVPR 2025): 250 h of fingerspelled phrases, 147 Deaf signers, phone video | Kaggle `googleai/fsboard`, size not confirmed (very large) | CC BY 4.0 | not planned: phrases without letter timing |
| `semlex` | Sem-Lex Benchmark: 84k isolated-sign videos, 41 Deaf signers | instructions at `github.com/leekezar/SemLex` | CC BY-NC-SA 4.0 | optional; letter glosses not checked |
| `surrey_fs` | Surrey/Exeter ASL finger-spelling, dataset A: 5 signers, 24 static letters, Kinect RGB + depth stills | direct link listed on the Exeter page (not checked) | none stated | optional supplementary static |
| `wlasl` | WLASL (YouTube/ASL-site URLs; many dead) | per-video download | **C-UDA 1.0**: computational/academic use only | P2 optional |
| `how2sign` | How2Sign sentence clips | per-request download | **CC BY-NC 4.0** | ingest script ported (phrases only) |
| `islr` | Kaggle "Google – Isolated Sign Language Recognition" (Holistic landmarks only) | Kaggle | verify on the Kaggle page | optional |
| ~~`PYTHON/Data`~~ | 26 folders, 10,676 cvzone crops with skeleton drawn on, 1 signer | n/a | n/a | **Rejected**: MediaPipe detects a hand in <1% (1/255 sampled); a baked-in overlay and near-duplicate frames made the old 99.9% meaningless |

## Vocabulary

- Tier 1: A–Z. 24 static letters go to the static model. **J and Z are dynamic** and are not recognised or scored
  yet: no source has multi-signer motion data for them (they can be recorded as 3 s takes). Built from video, see
  [PLAN.md §7](PLAN.md#7-alphabet-from-video).
- Tier 2: 50 words, [PLAN.md §4](PLAN.md#4-proposed-vocabulary-ask-1) (`backend/app/vocabulary.py`). A word stays if a
  source has ≥15 clips and ≥8 signers for it; `scripts.datasets.inspect_asl_citizen` reports this per word.

## Word pipeline (ASL Citizen)

```powershell
python -m scripts.datasets.inspect_asl_citizen                 # index + split CSVs only (~10 MB), no videos
python -m scripts.datasets.ingest_asl_citizen --dry-run        # what would be fetched, and how many MB
python -m scripts.datasets.ingest_asl_citizen --word goodbye=BYE --max-gb 1 --workers 3   # more workers run out of RAM
python -m scripts.build_reference_videos --category word
python -m scripts.dataset_report --write-docs                  # regenerates the table below
```

Glosses are matched by letters only (`THANK-YOU` = `THANKYOU`, `HELLO2` is a variant of `HELLO`). The variant with the
most clips is chosen. Synonym glosses (e.g. `TOILET` for *bathroom*) are only used when you pass
`--word bathroom=TOILET`, since they can be different signs. Reference videos come from **training-split** signers.

## Counts

<!-- counts:start -->
Generated 2026-10-27 by `python -m scripts.dataset_report --write-docs`. Signers are counted per source (`source:signer`).

| Category | Source | Signs | Clips | Signers | Clips by split |
|---|---|---|---|---|---|
| alphabet | chicago_fsvid | 24 | 3780 | 4 | none 3780 |
| alphabet | kaggle_asl_alphabet | 24 | 7374 | 1 | none 7374 |
| word | asl_citizen | 50 | 1554 | 48 | test 633, train 737, val 184 |

Words with ≥15 clips and ≥8 signers: **50 of 50**.
Letters with clips: 24 of 26; fewest signers for any letter: 0.
<!-- counts:end -->
