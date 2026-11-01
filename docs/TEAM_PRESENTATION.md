# Team presentation: Sign-ify

A 10 to 12 minute class presentation by five presenters: A, K, S, Y and J. Replace the initials with your names in
the scripts (presenter J is a person; the letter J is a handshape). Each presenter has the files they own, a 2-minute
script with demo cues, three likely questions with honest answers, a demo checklist and one slide.

This file describes the app as it is in this working copy: Sign → text has two live modes (letters and words), and
the letters J and Z are neither recognised nor scored.

## Ground rules

- **Numbers.** Say only the numbers in the [fact sheet](#fact-sheet). Each one is copied exactly from
  [EVALUATION.md](EVALUATION.md) or `backend/models/*/metrics.json`. Don't round or convert them: say 81.9% as
  written, and give an AUC as written (0.994), not as a percentage. If someone asks for a number that isn't there,
  say we didn't measure it.
- **Limits.** Each of the [five limits](#limits-we-say-out-loud) is said out loud by the presenter it belongs to, and
  J repeats all five at the end. Say them plainly; don't soften them.
- **Keep two things off the screen.** `docs/EVALUATION.md` was generated before J and Z scoring was taken out of the
  code. Its section "J and Z (letters that move)", and the end of its alphabet limitation ("they are scored against
  one reference signer's video"), no longer match the app: in this version, practising J or Z shows the reference
  video and says the letter "isn't scored yet". `README.md` has an unresolved merge conflict. Don't show or quote
  either.
- **Questions.** The presenter who owns the topic answers. If nobody knows, say so.

## Built with an AI coding assistant

Sign-ify was built with an AI coding assistant. J says so in the closing lines, and anyone who is asked should
confirm it. That makes it more important, not less, that each presenter can explain their own part without notes:

- what each file you own does, and why it was built that way;
- where each of your numbers comes from, and what it means;
- what your part cannot do.

Before the day, read your files and the doc that goes with them, and run your tests. Then explain your part to
another member without notes and answer their hardest questions. If you are asked who did what, be specific and
truthful: what you designed or decided, what you wrote or changed, and what the assistant generated and you
reviewed. Don't claim more than you did.

## Run order

| Presenter | Part | Live demo |
|---|---|---|
| A | Opening; how the camera sees hands | Sign → text, Letters (live) |
| K | The data and its licenses | Practice > Alphabet > A; Record dataset |
| S | The two recognition models and how they were tested | Sign → text, Letters (live) and Words (live) |
| Y | The practice coach | Practice > Words > name; Progress |
| J | Text → sign; how it all fits together; the limits; close | Text → sign |

Five parts of about two minutes make 10 minutes; the rest of the 12 is for handovers and demo trouble. If you run
late, cut a demo, never a limit.

Two shared slides frame the talk:

- **Title slide**, shown while A introduces the team: "Sign-ify: an ASL learning coach", your five names, the line
  "A learning aid, not a translator", and a screenshot of the app header with its tabs.
- **Closing slide**, which J puts up for the closing lines and leaves up during questions: the
  [five limits](#limits-we-say-out-loud), one line each.

## Fact sheet

The only numbers we say. "Written in" is where to point if someone asks.

**Words model** (`words-20261028-184044`, a CNN, tested on signers it never saw)

| Say | What it is | Written in |
|---|---|---|
| 81.9% | top-1 accuracy: its first guess is right | EVALUATION.md, Words; metrics.json `test.accuracy`: 0.8189 |
| 94.3% | top-5 accuracy: the right word is among its first five | EVALUATION.md; `test.top5_accuracy`: 0.9427 |
| 0.818 | macro-F1 | EVALUATION.md; `test.macro_f1`: 0.8183 |
| 63.0% / 81.4% / 94.4% | accuracy per signer, min / mean / max | EVALUATION.md |
| train 32 / val 5 / test 11 signers; clips 750 / 188 / 646 | the split by person | EVALUATION.md |
| 0.023 | ECE (gap between confidence and real accuracy) after temperature scaling | EVALUATION.md; `test.ece`: 0.0229 |
| 0.65 (val coverage 81.4%) | below this confidence the app says "Not sure yet" | EVALUATION.md |
| cnn 0.880, bigru 0.860 | validation macro-F1, which chose the CNN | EVALUATION.md, Candidates |
| cnn 81.9%, bigru 83.4% | test top-1 of both candidates, shown for transparency | EVALUATION.md, Candidates |
| good 0.38, happy 0.41, thank you 0.48 | weakest signs (test F1) | EVALUATION.md |
| good → thank you 33%, thank you → happy 33%, good → happy 25% | most frequent confusions | EVALUATION.md |
| family, finish, name, need, office, teacher: `"f1": 1.0`; hello: 0.857 | test F1 per word (pick demo words here) | metrics.json `test.per_class` |

**Alphabet model** (`alphabet-20261027-175212`, an MLP, 24 static letters)

| Say | What it is | Written in |
|---|---|---|
| 75.0% | test accuracy on one held-out signer (chicago_fsvid:rita), protocol signer_holdout | EVALUATION.md, Alphabet; `test.accuracy`: 0.75 |
| 0.659 | macro-F1 | EVALUATION.md; `test.macro_f1`: 0.6593 |
| 96.0% | top-5 accuracy | EVALUATION.md; `test.top5_accuracy`: 0.9596 |
| 0.036 | ECE | EVALUATION.md; `test.ece`: 0.0358 |
| mean 77.1% (andy 79.5%, drucie 84.9%, rita 75.4%, robin 68.5%) | leave-one-signer-out | EVALUATION.md |
| "verification needs ≥3 held-out full-frame video signers (has 0)" | why it is UNVERIFIED | EVALUATION.md; `"verified": false`, `"full_frame_signers": []` |
| D → K 51%, O → N 30%, I → T 27% | most frequent confusions | EVALUATION.md |
| mlp 0.7615, rf 0.732 | validation macro-F1, which chose the MLP (test accuracy: mlp 0.75, rf 0.7608) | metrics.json `candidates` |
| C 0.949, F 0.938, L 0.917, B 0.886; D 0.4 | test F1 per letter (demo letters, and one to avoid) | metrics.json `test.per_class` |
| `"excluded_dynamic": ["J", "Z"]` | J and Z are not in the model | metrics.json |

**Practice scores**

| Say | What it is | Written in |
|---|---|---|
| AUC 0.994 (classifier probability alone 0.993; 2584 pairs) | how well the word practice score ranks a correct sign above a different sign, on test signers | EVALUATION.md, Practice score; `scoring.test_auc`: 0.9937 |
| handshape 0.81, orientation 0.79, location 0.76, movement 0.63, two_hand 0.71 | the same AUC for each component on its own | EVALUATION.md |
| 5 validation signers | what the score's weights were fitted on | EVALUATION.md, Limitations |
| DTW vs 5 references | real recordings each word attempt is compared with, from training signers only | EVALUATION.md, Latency; metrics.json `references` |
| AUC 0.827 (93 repeat / 655 different-sign pairs); median score 91.1 vs 30.0 | custom-sign scoring, in a proxy test on held-out ASL Citizen signers | EVALUATION.md |
| a median of ~94 (26 letters of one signer) | why letters are never scored against a single recording | EVALUATION.md, Limitations |

**Data** (EVALUATION.md, Data)

| Source | Signs | Clips | Signers |
|---|---|---|---|
| asl_citizen (words) | 51 | 1584 | 48 (clips: test 646, train 750, val 188) |
| chicago_fsvid (letters) | 24 | 3780 | 4 |
| kaggle_asl_alphabet (letters, training only) | 24 | 7374 | 1 |
| wikimedia_commons (letter reference videos) | 26 | 26 | 1 |

Also there: "Letters with clips: 26 of 26; fewest signers for any letter: 1."

**Speed** (EVALUATION.md, Latency, "measured on this machine")

| Say | What it is |
|---|---|
| 37.3 ms | features for one clip (Python; ~58 frames) |
| 18.6 ms | word classification in JavaScript (Node, same code as the browser) |
| ~60 ms/frame per process (hand ≈33, pose ≈22) | MediaPipe landmark extraction, offline |
| not measured | browser FPS and end-to-end live latency |

## Limits we say out loud

| Limit | Say it plainly | Evidence, quoted exactly | Who |
|---|---|---|---|
| Alphabet accuracy is unverified | "The letter model's 75.0% is unverified: it comes from one held-out signer in studio hand crops, not from webcam video." | EVALUATION.md: "UNVERIFIED: verification needs ≥3 held-out full-frame video signers (has 0)" and "not webcam accuracy". metrics.json: `"verified": false` | S |
| No webcam validation | "Nobody has measured how accurate or how fast the app is with a live webcam." | EVALUATION.md: "Browser FPS and end-to-end live latency were not measured: no browser with a webcam was available in this environment." and "Accuracy is for ASL Citizen's webcam recordings; your camera, lighting and framing can lower it." | A (speed), S (accuracy) |
| J and Z are not recognised | "J and Z are drawn in the air, and we have only one signer's video of them, so the app doesn't recognise or score them." | metrics.json: `"excluded_dynamic": ["J", "Z"]`. EVALUATION.md: "J and Z are not recognised live" and "fewest signers for any letter: 1" | S, Y |
| Non-commercial dataset licenses | "Our word dataset is for non-commercial research only, and our main letter dataset has no license at all, so Sign-ify can't be sold or released publicly." | EVALUATION.md: "Several datasets are non-commercial or research-only (docs/DATASET.md). Don't deploy publicly or commercially with them." | K |
| Word-by-word glossing, not ASL grammar | "Text to sign is word by word, not ASL grammar." | EVALUATION.md: "Text → sign is word-by-word glossing; ASL grammar, word order and non-manual markers (facial grammar) are not modelled." | J |

The overall framing, which A gives in the opening, is EVALUATION.md's first limitation: "A learning aid, not a
translator."

---

## A: opening, and how the camera sees hands

**Files A owns**

- Browser: `frontend/src/lib/vision.js` (MediaPipe hand and pose trackers), `frontend/src/lib/features.js` (the
  feature recipe), `frontend/src/lib/segmenter.js` (when a sign starts and ends), `frontend/src/components/Camera.jsx`
  (camera, drawing, warnings), `frontend/src/components/SkeletonPlayer.jsx`, `frontend/scripts/features_cli.mjs`,
  `frontend/scripts/segmenter_cli.mjs`
- Server: `backend/app/features.py` (the same recipe in Python), `backend/app/extract.py` (MediaPipe on video files),
  `backend/app/holds.py` (still moments in letter videos), `backend/app/mediapipe_models.py`,
  `backend/scripts/download_models.py`, `backend/models/mediapipe/`
- Docs: [FEATURE_SPEC.md](FEATURE_SPEC.md)
- Tests: `backend/tests/test_features.py`, `test_feature_parity.py`, `test_segmenter.py`, `test_holds.py`,
  `test_extract.py`, and the helper `synth.py`

**Script (about two minutes)**

> Hi, we're A, K, S, Y and J, and this is Sign-ify, a web app for learning American Sign Language. You can sign to
> your webcam and see what it recognises, practise signs with feedback, and type English to see the signs. It's a
> learning aid, not a translator.
>
> I'll start with how the computer sees your hands. *[Demo: Sign → text, Letters (live). Raise your hand.]* All of
> this runs in the browser. Google's MediaPipe finds points on each hand, like the wrist, knuckles and fingertips,
> plus points on the face and shoulders. Your dominant hand is drawn in yellow. To recognise or score a sign, only
> these points go to our server, never the video.
>
> Raw points change if you simply step closer, so we turn them into features that don't: the shape of the hand,
> which way the palm faces, where the hand is compared with your forehead, mouth, chin or chest, measured in shoulder
> widths, and how it moves. We wrote this recipe twice, in JavaScript for the browser and in Python for training,
> and a test checks that both give the same numbers.
>
> The camera also decides when a sign starts and ends. *[Demo: hold the letter C still.]* A letter is captured when
> your hand holds still; a word, from when you start moving until you pause. *[Demo: move your hand to the edge of
> the frame.]* And it warns you when something's off, like now: my hand is too close to the edge.
>
> One gap: we never measured speed in a real browser. Our only timing is offline: 37.3 ms to compute one clip's
> features in Python.
>
> K, where does our data come from?

**Likely questions**

- **Q: Why work with hand points instead of the video itself?**
  Answer: The points are small enough to process live in the browser, and the same MediaPipe models and the same
  feature recipe handle both the dataset videos and your webcam, so training data and live input look alike. To
  recognise or score a sign, the video never leaves the browser; only "Add a sign" and "Record dataset" save video, on
  the computer running the server. The cost: if MediaPipe misses or misplaces a hand (poor light, blur, overlapping
  hands, a hand out of frame), we have nothing to fall back on. And we use face points only to locate the hand; we
  don't read facial expressions.
- **Q: Does it work in any lighting, for anyone, for left-handed signers?**
  Answer: We don't know how well, because nobody has measured it with a live webcam. The camera helps: it warns when
  it's too dark, when a hand is near the edge, when no hand is found, when the frame rate is low, and, for words, when
  your face and shoulders are out of view. Left-handed signers tick "Left-handed" in the header, and their features
  are mirrored.
- **Q: How fast is it?**
  Answer: In a browser, we don't know: EVALUATION.md says browser FPS and end-to-end live latency were "not
  measured". The camera shows its frame rate live. Our only timings are offline, on our machine: 37.3 ms to compute
  one clip's features in Python, and ~60 ms/frame per process for MediaPipe extraction.

**Demo checklist**

- [ ] Before you start talking: the app is running (`run_all.bat`), Sign → text > Letters (live) is open, the camera
  is allowed, and the landmarks already show.
- [ ] Stand with your face, shoulders and hands in frame, with light in front of you.
- [ ] Raise your hand: point at the yellow dominant hand and the white face and shoulder points.
- [ ] Hold C still until it is captured ("Last auto-captured: C" appears under Top matches).
- [ ] Move your hand to the edge until "Hand near the edge of the frame" appears.
- [ ] If tracking hasn't started after one retry, play your backup video and keep talking.

**Slide**

- **Title:** How Sign-ify sees your hands
- **Bullets:**
  - MediaPipe runs in the browser and finds points on both hands, the face and the shoulders; recognition and
    scoring use only these points, never the video
  - The points become features that don't change when you move: handshape, palm direction, location relative to the
    face and chest (in shoulder widths), and movement; one recipe in JavaScript and Python, checked by a parity test
  - Auto-capture: a letter on a still hold, a word from first movement to pause; live speed not measured (offline
    features: 37.3 ms per clip)
- **Screenshot:** Sign → text > Letters (live) with your hand raised: the landmarks drawn (yellow dominant hand, white
  face and shoulder points) and the camera's frame-rate readout.

---

## K: the data and its licenses

**Files K owns**

- Dataset scripts: `backend/scripts/datasets/` (`asl_citizen.py`, `common.py`, `inspect_asl_citizen.py`,
  `ingest_asl_citizen.py`, `ingest_chicago_fsvid.py`, `ingest_alphabet_images.py`, `ingest_commons_alphabet.py`,
  `ingest_how2sign.py`), `backend/scripts/dataset_report.py`, `build_reference_videos.py`,
  `build_reference_images.py`, `record_cli.py`
- Storage and database: `backend/app/remote_zip.py` (reads single clips out of the online archive),
  `backend/app/clipstore.py` (one clip = one landmark file), `backend/app/db.py`, `backend/migrations/`,
  `backend/alembic.ini`, `backend/app/registry.py`, `backend/app/vocabulary.py` (the word list)
- Browser: `frontend/src/pages/Record.jsx` (the dataset recorder)
- Docs: [DATASET.md](DATASET.md)
- Tests: `backend/tests/test_asl_citizen.py` (with the helper `fake_citizen.py`), `test_commons_alphabet.py`,
  `test_remote_zip.py`, `test_migrations.py`, `test_reference_media.py`

**Script (about two minutes)**

> Thanks, A. A model is only as good as its examples, so my part is the data: where it comes from, how we store it,
> and what we're allowed to do with it.
>
> We only use real people signing. For the words, we use ASL Citizen, a Microsoft research dataset of people signing
> to their own webcams. We took 1584 clips of our 51 words, by 48 different signers, and our script pulls just those
> clips out of the online archive instead of downloading the whole thing.
>
> Letters were harder. ChicagoFSVid gave us 3780 clips of 24 letters from 4 signers. A Kaggle photo collection added
> 7374 clips, but all from 1 person, so we train on them and never test on them. And a Wikimedia Commons video of one
> signer spelling all 26 letters gives every letter a reference video.
>
> Each clip is stored as landmark points, plus a database record of its source, signer, split and license. The split
> is the key: we split by person, so nobody we test on appears in training. For words, that's 32 signers to train, 5
> to validate and 11 to test.
>
> *[Demo: Practice > Alphabet > A.]* That's also why every reference shows who signed it and its license, right
> under the video.
>
> And licenses are a real limit. ASL Citizen is for non-commercial research only, and ChicagoFSVid states no license
> at all, so we treat it as research-only too. Sign-ify can't be sold or released publicly with this data.
>
> *[Demo: Record dataset.]* Last, our recorder collects new data with a countdown and a quality check, and we only
> record people who agree.
>
> S will show what we trained on all this.

**Likely questions**

- **Q: Why not just use sign videos from YouTube or the web?**
  Answer: Those videos belong to the people who posted them, and the signers never agreed to be in a training set.
  We only used datasets whose terms allow research use, and we record our own data only with consent; the recorder
  page says "Record only people who have agreed." More data would help most for the letters.
- **Q: Could you publish Sign-ify or sell it?**
  Answer: Not with this data. ASL Citizen's license is "non-commercial, non-revenue research use only" (it's in the
  caption under every word video), and ChicagoFSVid states no license, so we treat it as research-only. EVALUATION.md
  says: "Don't deploy publicly or commercially with them." The trained models come from that data too. One thing we
  still have to fix: the trained model files are in our public GitHub repository, which those terms may not allow;
  the licenses section of [DEPLOYMENT.md](DEPLOYMENT.md) explains it.
- **Q: Are four signers, or one, enough for the letters?**
  Answer: No, and we say so. ChicagoFSVid has 4 signers, the Kaggle photos 1, and J and Z only 1: EVALUATION.md
  lists "fewest signers for any letter: 1". That's why the alphabet is marked UNVERIFIED and J and Z aren't
  recognised. For words we had 48 signers, enough to keep 11 apart for testing.

**Demo checklist**

- [ ] Practice > Alphabet > A: point at the caption under the video (Wikimedia Commons, CC BY-SA) and the one under
  the handshape photo (Kaggle, GPL).
- [ ] Practice > Words > name: the caption ends "non-commercial, non-revenue research use only".
- [ ] Record dataset: signer id `demo_k`, click B, Start session, then Record (Space). After the countdown, show the
  quality line ("Hands … · face … · … frames"), then press Esc so nothing is saved (Space would save it).
- [ ] Don't open `docs/EVALUATION.md` or `README.md` on screen (see [Ground rules](#ground-rules)).
- [ ] If anything fails after one retry, play your backup video.

**Slide**

- **Title:** Real signers, split by person, licensed for research
- **Bullets:**
  - Words: ASL Citizen, 1584 clips of 51 words by 48 signers. Letters: ChicagoFSVid (3780 clips, 4 signers), Kaggle
    photos (7374 clips, 1 signer, training only), Wikimedia Commons (26 letters, 1 signer, reference videos)
  - Split by person: nobody we test on appears in training (words: train 32 / val 5 / test 11 signers)
  - ASL Citizen: non-commercial research only; ChicagoFSVid: no license stated. So: no public or commercial release
- **Screenshot:** Practice > Alphabet > A: the reference video with its attribution and license caption, and the
  handshape photo with its license caption.

---

## S: the two recognition models, and how they were tested

**Files S owns**

- Training and evaluation: `backend/app/train_static.py` (letters: random forest vs neural network),
  `backend/app/train_words.py` (words: CNN vs BiGRU), `backend/scripts/train_alphabet.py`,
  `backend/scripts/train_words.py`, `backend/scripts/evaluation_report.py`, `backend/requirements-train.txt`
- Models at run time: `backend/app/recognition.py`, `backend/app/word_model.py` (shared: Y explains its DTW scoring
  half), `backend/models/alphabet/` and `backend/models/words/` (each version with its `metrics.json`)
- Browser: `frontend/src/lib/staticModel.js`, `wordModel.js`, `wordWorker.js`, `useLiveLetters.js`,
  `useLiveWords.js`, and `frontend/src/pages/Recognize.jsx` (Sign → text)
- Docs: [EVALUATION.md](EVALUATION.md)
- Tests: `backend/tests/test_training.py`, `test_words.py`, `test_integration_video.py`

**Script (about two minutes)**

> Thanks, K. I'll show the two models that turn A's features into answers, and how we tested them.
>
> *[Demo: Sign → text > Letters (live). Sign C, then L.]* The letter model reads the shape of a still hand. We tried
> a random forest and a small neural network, and kept the network because it did better on our validation signers.
> It knows 24 letters. J and Z are left out: they're drawn in the air, and we have only one signer's video of them,
> so they aren't recognised.
>
> *[Demo: Words (live). Sign "name", then pause.]* The word model reads movement over time. It's a small
> convolutional network for our 51 words. On 11 people it never saw in training, its first guess was right 81.9% of
> the time, and the right word was in its top five 94.3% of the time. It depends a lot on the person: from 63.0% for
> the hardest signer to 94.4% for the easiest. Its weakest words are "good", "happy" and "thank you", which it mixes
> up with each other.
>
> Both models run in your browser, and when you finish a sign, the server checks it again. The confidence is
> calibrated, so when the model isn't sure, it says "Not sure yet" instead of guessing.
>
> Now the honest part. *[Point at the header badges.]* The letter model shows 75.0%, but it's marked unverified.
> That number comes from one held-out signer in studio hand crops, not webcam video. Our rule needs at least 3
> held-out full-frame video signers, and for letters we have 0. And nobody has measured accuracy with a live webcam.
>
> Y, how do we turn this into a coach?

**Likely questions**

- **Q: Will it be that accurate on my webcam?**
  Answer: Maybe not, and we haven't measured it. The word test clips are webcam recordings by 11 ASL Citizen signers
  the model never saw, which is the closest we have, but they weren't recorded through our app, and accuracy varies a
  lot by person (63.0% to 94.4%). EVALUATION.md warns: "your camera, lighting and framing can lower it." The letter
  test isn't webcam video at all.
- **Q: The BiGRU scored 83.4% on test, above the CNN's 81.9%. Why didn't you use it?**
  Answer: We pick models on validation signers only, before looking at the test: CNN 0.880 against BiGRU 0.860
  validation macro-F1. Choosing whatever wins on the test set would make the test score an optimistic guess instead
  of an honest one, so we publish both test numbers for transparency instead. The same happened with letters: the
  neural network won on validation (0.7615 against 0.732), while the random forest was slightly better on test
  (0.7608 against 0.75).
- **Q: Why can't it recognise J and Z?**
  Answer: J and Z are traced in the air, and the letter model only looks at a still handshape; metrics.json lists
  them under "excluded_dynamic". Learning the movement needs several signers' recordings, and we have one signer's
  video ("fewest signers for any letter: 1"). The app shows their reference video, but doesn't recognise or score
  them.

**Demo checklist**

- [ ] Before the talk, open Sign → text > Words (live) once so the word model is already loaded.
- [ ] Letters (live): sign C (test F1 0.949), then L (0.917). Avoid D, E, I, N and O: they are the common
  confusions.
- [ ] Words (live): sign "name" or "family" (test F1 1.0), then pause. Show the top five and the "server check" line.
- [ ] Point at the header badges: "Alphabet: 75.0% (unverified)" and "Words: 81.9% (held-out signers)".
- [ ] Don't try J, Z, "good", "happy" or "thank you".
- [ ] If recognition fails after one retry, play your backup video.

**Slide**

- **Title:** Two models, tested on people they never saw
- **Bullets:**
  - Words (CNN, 51 words): top-1 81.9%, top-5 94.3% on 11 held-out signers; per signer 63.0% / 81.4% / 94.4% (min /
    mean / max)
  - Letters (MLP, 24 static letters): 75.0% on one held-out signer, UNVERIFIED: studio hand crops, "not webcam
    accuracy"; J and Z not recognised
  - Models chosen on validation signers only; calibrated confidence (ECE 0.023 for words), so it says "Not sure yet"
    instead of guessing
- **Screenshot:** Sign → text > Words (live) after signing "name": the top matches with confidence bars, with the
  header badges "Alphabet: 75.0% (unverified)" and "Words: 81.9% (held-out signers)" in view.

---

## Y: the practice coach

**Files Y owns**

- Scoring and hints: `backend/app/clip_scoring.py` (custom signs: comparison with your own recording),
  `backend/app/content.py` (handshape descriptions and tips), `backend/scripts/calibrate_own_clip.py`. Y also
  explains the DTW scoring half of `backend/app/word_model.py` (owner: S) and the score, attempts, progress and
  custom-sign endpoints in `backend/app/main.py` (owner: J).
- Browser: `frontend/src/pages/Practice.jsx`, `frontend/src/pages/Progress.jsx`, `frontend/src/pages/Library.jsx`
  (Add a sign), `frontend/src/components/ReferencePanel.jsx`, `frontend/src/lib/identity.js` (anonymous learner id
  and settings)
- Tests: `backend/tests/test_clip_scoring.py`, `test_delete_sign.py`, `test_rerecord_sign.py`

**Script (about two minutes)**

> Thanks, S. My part is the practice coach: helping you get better, not just telling you what you signed.
>
> *[Demo: Practice > Words > name.]* You pick a sign and watch a real person sign it, slowed down if you like. Then
> you sign it, pause, and get a score.
>
> For a letter, the score is how sure the letter model is that you made the right letter. If it looks like a
> different one, you get a tip, like "That looks more like S. Move your thumb to the side of your index finger, not
> across the front." And the joints that differ most turn red on your hand.
>
> For a word, we combine the word model's confidence with a comparison against 5 recordings by real signers. We line
> your attempt up with theirs in time and compare handshape, palm orientation, location, movement and the use of two
> hands. *[Point at the bars and hints.]* That's where these bars and hints come from, like "Your hand is too low".
>
> Does the score work? On held-out signers, it ranks a correct attempt above a different sign with an AUC of 0.994.
> But the classifier alone gets 0.993, so the parts mostly explain the score rather than improve it, and the weights
> come from just 5 validation signers. It's an estimate.
>
> *[Demo: press R, then open Progress.]* You can replay your attempt next to the reference. Progress keeps your
> scores without an account, marks signs as mastered, and brings tricky signs back sooner.
>
> You can also add your own sign, like a name sign; it's scored against your own recording. In a proxy test, someone
> repeating a word scored a median of 91.1, and a different word 30.0, but that's unverified on custom signs. And J
> and Z aren't scored yet.
>
> J will finish with text to sign.

**Likely questions**

- **Q: Can I trust the score like a teacher's mark?**
  Answer: No; it's an estimate. On held-out signers it ranks a correct attempt above a different sign with AUC
  0.994, but the classifier alone gets 0.993, the weights were fitted on 5 validation signers, it has never been
  compared with a teacher's marks, and nobody has measured it with a live webcam. The score card itself says "Scores
  are a learning aid, not a certification."
- **Q: How is a custom sign scored, and how reliable is that?**
  Answer: Your attempt is lined up in time with your own recording and compared part by part. We tested the idea on
  held-out ASL Citizen signers: the same person repeating a word, against the same person making a different word,
  gave AUC 0.827, with median scores of 91.1 against 30.0. It's unverified on custom signs, and a sign with no
  recording can't be scored at all.
- **Q: What does the app keep about me?**
  Answer: There's no account: the browser keeps an anonymous learner id. In Practice, only your scores are saved, in
  the database on the computer running the server; your hand points are used to score you and are not kept. Video is
  saved only by "Add a sign" and "Record dataset", on that same computer. Deleting a custom sign removes its files;
  its past scores stay in Progress, marked as deleted.

**Demo checklist**

- [ ] Earlier that day, practise a few signs in this browser so Progress isn't empty.
- [ ] Practice > Words > name: slow the reference video down, sign once, pause. Show the score, the bars and the
  hints.
- [ ] Press R: your attempt replays next to the reference landmarks.
- [ ] Open Progress: the new attempt, the mastered count, "Review now". If asked for the mastery rule, read it from
  that page.
- [ ] Optional, if there's time: Practice > Alphabet > A, make an S on purpose to show a tip and red joints (the tip
  depends on the letter the model thinks it sees).
- [ ] If anything fails after one retry, play your backup video.

**Slide**

- **Title:** A coach, not just a recogniser
- **Bullets:**
  - Watch a real signer, sign, then get a score, bars for handshape, orientation, location, movement and two hands,
    hints, and red joints where your hand differs
  - Word score: the classifier plus DTW against 5 real recordings; AUC 0.994 on held-out signers (classifier alone
    0.993); weights from 5 validation signers, so it's an estimate
  - Progress without an account; custom signs scored against your own recording (proxy AUC 0.827, unverified); J and
    Z not scored
- **Screenshot:** Practice > Words > name just after an attempt: the reference video on the left, the score card with
  its component bars and hints on the right.

---

## J: text → sign, how it all fits together, and the close

**Files J owns**

- Text → sign: `backend/app/text_plan.py`, `frontend/src/pages/TextToSign.jsx`,
  `frontend/src/components/VideoSequencePlayer.jsx`, `frontend/src/components/SentencePractice.jsx`,
  `frontend/src/components/SpeedControl.jsx`, `frontend/src/lib/playback.js`, `frontend/src/lib/speech.js`
- The app around it: `backend/app/main.py` (the API; Y explains the scoring and progress endpoints),
  `backend/app/schemas.py`, `backend/app/config.py`, `backend/app/storage.py`, `frontend/src/App.jsx`,
  `frontend/src/api.js`, `frontend/src/main.jsx`, `frontend/src/styles.css`, `frontend/index.html`,
  `frontend/vite.config.js`, `frontend/package.json`
- Dev-only fallback, never used for real scoring: `backend/app/legacy/`, `backend/scripts/seed_dev_synthetic.py`
- Running and deploying: `run_all.bat`, `run_tests.bat`, `start_backend.bat`, `start_frontend.bat`,
  `backend/requirements.txt`, `backend/requirements-dev.txt`, `backend/requirements-deploy.txt`, `backend/Dockerfile`,
  `backend/scripts/publish_store.py`, `backend/.env.example`, `frontend/.env.example`
- Docs: `README.md`, [PLAN.md](PLAN.md), [DEPLOYMENT.md](DEPLOYMENT.md)
- Tests: `backend/tests/test_text_plan.py`, `test_api.py`, `test_playback_speed.py`, `test_storage.py`, and
  `conftest.py`

**Script (about two minutes; slightly longer because it closes the talk)**

> Thanks, Y. My part goes the other way, from English to signs, and then I'll close.
>
> *[Demo: Text → sign. Click the example "my mother is happy".]* The app looks up each word, or a short phrase like
> "thank you", and plays real signers' videos in order, with captions. You can slow them down, loop them, or step
> through with the arrow keys. A word we don't have, like "my", is fingerspelled letter by letter.
>
> With this box ticked, it also makes the English a little more like ASL: it drops words ASL usually leaves out,
> such as "the", "is" and "to", and turns "eating" into EAT. It always shows what it changed. *[Point at "Not signed:
> is".]*
>
> But this is word-by-word glossing, not translation. ASL has its own grammar, its own word order, and facial
> expressions that carry meaning, and we don't model any of that. It's still useful for learning: *[Demo: click
> "Practise the sentence".]* you can practise the whole sentence, sign by sign.
>
> How does it all fit together? Your browser runs the camera and the live models. A Python server stores signs and
> recordings, scores attempts, and builds these sequences. One script, run_all, starts both.
>
> So, the limits we want you to remember. *[Closing slide.]* The letter accuracy is unverified. Nobody has measured
> how the app does with a live webcam. J and Z are not recognised. The datasets we trained on include non-commercial,
> research-only ones, so Sign-ify can't be released publicly. And text to sign is word by word, not ASL grammar.
>
> We built Sign-ify with an AI coding assistant, and each of us can explain our part. Thank you; we're happy to take
> questions.

**Likely questions**

- **Q: Why doesn't it translate into real ASL?**
  Answer: Real ASL isn't English in signs: it has its own grammar, word order, use of space and facial grammar.
  Translating properly would need lots of paired English and ASL data, and fluent signers to check the output; we had
  neither. So we do honest word-by-word glossing and say so on the page. EVALUATION.md: "Text → sign is word-by-word
  glossing; ASL grammar, word order and non-manual markers (facial grammar) are not modelled."
- **Q: Is it online? Can I use it at home?**
  Answer: It runs locally on our laptop. We wrote a guide for free hosting ([DEPLOYMENT.md](DEPLOYMENT.md)), but we
  haven't deployed it, and with these datasets we shouldn't publish it: EVALUATION.md says "Don't deploy publicly or
  commercially with them."
- **Q: What happens with a word that isn't in the vocabulary, or a typo?**
  Answer: For longer words it tries a close spelling match, and with the gloss box ticked it tries base forms (eating
  → EAT); the page lists every change. Anything still unknown is fingerspelled. Digits have no sign here and are
  listed under "Not available".

**Demo checklist**

- [ ] Text → sign, with "Approximate ASL-style glosses" ticked (it is by default).
- [ ] Click the example "my mother is happy". Show "Not signed: is", the fingerspelled MY, then MOTHER and HAPPY
  playing; press → to step, tick loop, slow it down.
- [ ] Click "Practise the sentence" to show sentence practice (no need to sign).
- [ ] Optional: type a short name with no J or Z in it to show fingerspelling.
- [ ] Put up the closing slide for the limits and leave it up for questions.
- [ ] If anything fails after one retry, play your backup video.

**Slide**

- **Title:** Text → sign: word by word, and honest about it
- **Bullets:**
  - Looks up words and short phrases and plays real signers' videos in order (captions, slower speeds, loop, arrow
    keys); unknown words are fingerspelled
  - Optional ASL-style glosses drop "the", "is" and "to" and turn eating → EAT, showing every change; practise the
    sentence sign by sign
  - Not ASL grammar: "ASL grammar, word order and non-manual markers (facial grammar) are not modelled"
- **Screenshot:** Text → sign after clicking "my mother is happy": the "Not signed: is" line, the fingerspelled MY,
  and the player showing MOTHER.

---

## Before the presentation

### Tests

- [ ] From the repo root, run `run_tests.bat` (or, in PowerShell, `cd backend` then
  `.\.venv\Scripts\python.exe -m pytest`). Everything must pass. Some tests skip without Node on PATH, without the
  downloaded MediaPipe models, without torch, or without local data; know which ones skipped and why.
- [ ] Each presenter also runs their own test files (listed under their Files), for example
  `.\.venv\Scripts\python.exe -m pytest tests/test_text_plan.py` from `backend`, and can say what each one checks.

### Build

- [ ] In `frontend`, run `npm run build`; it must end with "built in".
- [ ] Start the app with `run_all.bat`, which opens the browser once the backend answers. Check that the header shows
  the signs count and both model badges, not "model: not trained".
- [ ] Click through every page used in a demo once.

### Backup video

- [ ] The day before, screen-record each presenter's demo on the presenting laptop (Snipping Tool's Record, or Xbox
  Game Bar with Win+Alt+R): one video per presenter, named after them.
- [ ] Copy the videos to the laptop's desktop and to a USB stick, and check they play with Wi-Fi off.
- [ ] Agree on the rule: if a live demo hasn't worked after one retry, play the video and keep talking.

### Room test

- [ ] Internet: the page downloads MediaPipe's tracking files from the internet when it loads. Test the room's Wi-Fi
  with the app; if it's blocked, use a phone hotspot or the backup videos.
- [ ] Camera: face, shoulders and hands in frame, light in front of the signer, a plain background, and no "Too dark"
  warning.
- [ ] Projector: mirror the screen, and zoom the browser until captions and score bars are readable from the back
  row.
- [ ] Laptop: plugged in, sleep and notifications off, other apps closed (the camera warns about a low frame rate).
- [ ] Don't click inside the backend window: Windows pauses a program while text is selected in its window (press Esc
  to resume).
- [ ] A full run-through in the room with a timer, handovers included, landing between 10 and 12 minutes.

### On the day

- Present from the laptop that has the local data store: the reference videos live there, not in git.
- Don't pull, stash or switch branches: demo the same working copy you rehearsed with (it has uncommitted changes).
- Keep `docs/EVALUATION.md` and `README.md` off the screen (see [Ground rules](#ground-rules)).
- Don't demo J, Z, D or E, or the words "good", "happy" and "thank you".
