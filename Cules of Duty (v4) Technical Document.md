# Team possession estimation: technical document

**Team:** `Cules of Duty` &nbsp;|&nbsp; **File to submit:** `Cules of Duty.csv` &nbsp;|&nbsp; **Repository:** (https://github.com/FutureVIPBoss/More-than-a-Hack)

---

## 1. Approach

The task is to label every one of the 199,319 frames of the tactical video `H` (home in possession), `A` (away), `D` (ball dead) or `X` (abstain). We treat it as **sequence decoding on top of the provided tracking**, not as a per-frame classification, because the ranking metric (spellF1) rewards getting possessions (their count, length and timing) right, and a model that flips label from frame to frame is punished even if most frames are right.

1. **Per-frame evidence from the tracking (ATD) and metadata.** For each frame: how strongly each team controls the ball (a distance-weighted sum over each team's tracked players), a smoothed nearest-player vote, team shape (convex-hull area of each team), player speed and the share of slow players, whether the ball is outside the pitch, how far the ball moves, and the Smart Tagging team and set-piece tags.
2. **Two small calibrated models turn the evidence into probabilities.**
   - *Home vs away*: a stack fitted to the **5 published ground-truth minutes** (scored out-of-fold, leave-one-minute-out). Its inputs are a weak-label model, the smoothed nearest-player vote, control evidence, team-shape difference and a Smart Tagging team prior. The weak-label model is a Linear Discriminant Analysis trained on the whole match with the Smart Tagging team tags as weak labels (39 geometric features, with home/away-swapped copies so it learns evidence rather than "the home team usually has the ball"; cross-fitted by 90-second blocks and Platt-calibrated; cross-fitted AUC against the weak labels 0.887). Six stack variants were compared (the weak-label model alone, a logistic stack, three ablations of it, and a nonlinear stack); the one used is the **nonlinear stack**: an Extra Trees classifier (200 trees, minimum leaf 20) trained on home/away-symmetrised copies (every input negated with the label flipped), window AUC 0.951 against 0.937 for the logistic stack.
   - *Dead vs alive*: three cue sets were compared: a linear model on player-speed, slow-player share, team spread and the set-piece tag (`relative`); the same plus ball cues (`ball_relative`); and an Extra Trees classifier on the `ball_relative` cues (`learned`), cross-fitted leave-one-minute-out and Platt-calibrated. Out-of-fold AUC: 0.719, 0.827 and 0.907. The model that was finally selected uses `relative` (see step 4 for why the best-scoring `learned` candidate was not selected).
3. **A three-state decoder with a compared emission design.** Each frame gets log-probabilities for `H`, `A`, `D` from the two models (each tempered because neighbouring frames are near-duplicates), and the best path is found with a cost per switch and a cost for entering dead ball (Viterbi), then a minimum run length (1 s possession, 2 s dead ball). Three ways of combining the two models were compared: *hierarchical* (`dead = log p(dead)`, `home = log p(alive) + log p(home | alive)`), *balanced* (alive vs dead decided by p(dead) alone, only the less likely team penalised) and *mixed* (hierarchical where the ball is tracked, balanced where it is missing). Each half is decoded separately; the stretch of the second half before kickoff (graded but dead, per the FAQ) is fixed to `D`.
4. **Selection.** The nine candidates (three emission designs × three dead-ball cue sets) each get a grid of 648 settings (tempering of each model in {0.1, 0.2, 0.4}, switch cost in {2, 4, 8}, dead cost in {6, 12, 20, 30}, dead-ball offset in {-1 … +2}). Per candidate, the 30 settings with the best frame F1 on the labelled window are re-scored on spellF1 and the best of them is kept; a **nested** estimate is computed alongside (setting chosen by F1 on four minutes and scored on the fifth, repeated for every minute). A candidate's score is `0.5 × spellF1 + 0.5 × nested Accuracy`. Two rejections apply first: (a) whole-match structure outside sanity bounds (0.55-2.1 dead stretches per minute, dead stretches of 8-45 s, 1.0-4.5 possessions per minute of 8-40 s); (b) a window operating point that disagrees by more than 1.0 with the dead shift needed to reach the assumed whole-match dead share. All nine candidates passed rule (a); seven failed rule (b), so two remained, and the winner is **mixed + relative** (tempering 0.20 for home/away and 0.40 for dead, switch cost 8, dead cost 30, window dead-ball offset +1.0). **The highest-scoring candidate on the window, mixed + learned (nested Accuracy 0.803, spellF1 0.523), was rejected by rule (b)** (its whole-match shift was -1.90 against a window offset of -0.5), so the submitted model is deliberately not the best one on the labelled window: we preferred a model whose behaviour on the window and on the whole match agree.
   The whole-match dead share is **fitted on the decoded labels** to an assumed 38% (football has the ball dead for roughly 30-45% of the clock; the window itself is 20% dead). This is an assumption from general football knowledge, not from any data about this fixture [[CONFIRM: true]]. [[FILL: result of the sensitivity run at 30% and 45% (dead share, possessions per minute, window metrics), or delete this sentence]]
5. **Why this and not something else.** A frame classifier plus smoothing was our baseline reproduction (it matched the published baseline scores exactly) and it over-fragments possessions. Hand-set emission weights were our first decoder; fitting them to the real labels lifted every frame-level metric (see Section 4).

## 2. Assets: used, and tried-and-dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical tracking (ATD txt + xml) | **yes** | the core signal: player and ball positions, speeds, team of each track |
| Smart Tagging | **yes, partly** | team tags are a weak label for the home/away model and a weak prior; set-piece tags are a weak dead-ball cue. They are incomplete (59 set pieces in a whole match; only 45% of truly dead frames in the public window fall inside a set-piece tag), so they are *never* the only dead-ball signal |
| Public ground truth (5 min) | **yes** | to fit and validate the small calibration models (allowed by the rules); never the private ground truth |
| Homography + tactical video | **diagnostics only** | OpenCV overlays to inspect mistakes. A classical (motion x brightness) ball detector **was tried to recover ball gaps and dropped**: on 150 sampled frames it returned a candidate on 96% but only 35% of those were within 1 m of the tracked ball |
| Broadcast tracking | **tried, dropped** | we aligned the broadcast ball track to the tactical one by trajectory matching; **no window passed the acceptance test** (0 of 119), so it is not used |
| Kalman / RTS ball smoother | **tried, not used** | on real data short-gap interpolation fills only 0.1% of the graded frames: the missing ball is in long gaps (150 gaps, median 10.7 s, longest 222 s), which smoothing cannot repair. The submitted pipeline uses linear interpolation of gaps up to 0.6 s |
| Tabular model lab (original / SMOTE / undersampled data, tuned RF / GB / XGBoost) | **tried, not used** | trained on weak labels; its dead-ball model was poor (AUC 0.62 against its weak labels, 0.72 against the real labels), so it is not in the final model |
| AutoML (PyCaret) | **development only** | used once to compare estimator families for the weak-label home/away model on grouped folds (Linear Discriminant Analysis ranked first). **The repository does not depend on it**: no PyCaret, LightGBM or XGBoost code remains in the notebook, and both estimators are pinned (`PIN_ESTIMATORS`), so no model search runs when the file is reproduced |

## 3. Possession rules we implemented

- **Loose or contested ball.** When no player is within reach of the ball, or both teams are equally close, the evidence is ~0 and the decoder **keeps the last team** (the switch cost does the work). A possession changes hands only after *sustained* evidence for the other team, not on a single touch or deflection.
- **Challenges.** A brief contest does not flip the label (minimum run length and switch cost); a successful challenge flips it once the other team is in sustained control.
- **Dead ball.** A frame is dead when the evidence says play has stopped: players slow and standing, the ball outside the pitch, ball nearly stationary, or a set-piece tag, and the model is allowed to say "dead" even if the nearest-player evidence is confident (that is the point of the hierarchical probabilities).
- **Restarts.** There is no separate restart model; the first label after a dead stretch is whatever the team evidence supports when play resumes. Before the second-half kickoff all graded frames are `D`.
- **Abstention.** We label every graded frame; `X` is written only outside the graded blocks (40,462 frames). [[CONFIRM: that an `X` inside the graded blocks would cost more than a wrong label, per the organisers' evaluation notes]]

## 4. Development results (public 5-minute window, out-of-fold unless stated)

| configuration | Accuracy | Precision | Recall | F1 | spellF1 | possessions (truth 17) |
|---|---|---|---|---|---|---|
| organisers' baseline, +/-50 smoothing | n/a | n/a | n/a | 0.62 (macro) | 0.326 | 26 |
| organisers' baseline, +/-125 smoothing | n/a | n/a | n/a | 0.58 (macro) | 0.424 | 16 |
| hand-set emissions, Viterbi | 0.691 | 0.687 | 0.617 | 0.621 | 0.400 | 8 |
| weights fitted to the real labels (earlier version) | 0.780 | 0.789 | 0.709 | 0.725 | 0.486 | 11 |
| earlier version (v2: logistic stack, mixed + learned dead cues) | 0.797 | 0.788 | 0.776 | 0.780 | 0.543 | 11 |
| **final model** (nested estimate, `possession_pipeline_FINAL_v4.ipynb`) | **0.752** | **0.725** | **0.697** | **0.705** | **0.350** (plain) | 7 |
| submitted file, scored on the window (in-sample sanity check, whole-match decode with the dead shift) | n/a | n/a | n/a | 0.882 (macro) | 0.500 | 7 |

Per class (out-of-fold): home precision 0.778 / recall 0.904 / F1 0.836; away 0.799 / 0.794 / 0.796; dead 0.813 / 0.511 / 0.627. On the window, team accuracy on truly alive frames is 0.895 and alive-vs-dead F1 is 0.953.

The window holds only 17 possessions (bootstrap interval for spellF1 about 0.2 to 0.6), so differences of a few points are not significant. The final model scores below the previous version on the window, because the consistency rule of Section 1 removed the best-scoring candidate (see step 4). We therefore also checked whole-match structure: **dead share 0.38 (fitted, by construction); 68 dead stretches (0.64 per minute, mean 35.4 s); 137 possessions (1.29 per minute, mean 28.8 s)**, against about 3.4 possessions per minute of about 14 s inside the window. Label counts on the graded frames: H 62,982, A 35,775, D 60,100.

## 5. Limitations and what we would fix with more time

- **Only five minutes of true labels, and an easy five minutes.** All fitted weights come from one stretch with about four dead-ball stretches; the ball is visible on 88% of its frames, against 59% over the match. In an earlier version a stress test that re-masked the window with realistic ball-missing patterns lowered accuracy from 0.709 to 0.647. Expect lower scores on unseen frames.
- **Possessions are merged.** On the window the submitted model predicts 7 possessions of 33.2 s on average against 17 true possessions of 14.2 s, which is why plain spellF1 is low (0.350). Two selected settings sit at the edge of our grid (switch cost 8 and dead cost 30 are the largest values), so a wider grid could do better.
- **The selected model is not the best one on the window.** We preferred agreement between window and whole match over the highest window score (Section 1, step 4); the cost is about 5 points of nested accuracy.
- **Dead-ball recall is the weakest part** (0.51 out-of-fold). Without a ball, stoppages are inferred from player behaviour, which is noisy; Smart Tagging covers under half of true dead time. With more time: a dedicated stoppage detector trained on more labelled data, and restart-aware logic (who takes the throw-in or free kick).
- **The whole-match dead share is an assumption.** The 38% is taken from general football knowledge, not measured on this match; its effect on the labelled window is small, but we cannot verify its effect on the rest of the match.
- **Dead ball leans on ball visibility.** In the submitted labels 58% of ball-missing frames are dead against 24% of frames where the ball is seen. This may partly be true (stoppages often lose the ball) but we could not verify it with five minutes of labels.
- **Long ball gaps are unrepaired.** A second ball source (broadcast, video) did not pass our quality tests; a better alignment or a learned ball detector could help. No boundary-shift correction is applied; an earlier analysis found about 36% of errors within 1.5 s of a true transition.
- **Weak labels.** The weak-label model learns from automatic tags, so it cannot exceed their quality.

## 6. Reproduction

```bash
git clone <public repository URL> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pinned.txt        # python 3.11.18, numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, scikit-learn 1.4.2, matplotlib 3.7.5
# unpack the provided data bundle into data/ keeping its names (see data/README.md), or set MTAH_DATA_DIR
python reproduce.py --notebook possession_pipeline_FINAL_v4.ipynb   # runs the lean final notebook, writes "evaluation/results/Cules of Duty.csv"
python evaluation/validate_submission.py "evaluation/results/Cules of Duty.csv"   # quotes needed: the name contains spaces
```

What makes the file reproducible:

- **No search at reproduction time.** The two learned estimators (Linear Discriminant Analysis for the weak-label home/away model, Extra Trees with `min_samples_leaf=5`, 200 trees for the learned dead-ball cue) are pinned in `PIN_ESTIMATORS`; the selection among the team models, emission designs and settings is deterministic given the pinned library versions.
- **Pinned environment.** `requirements-pinned.txt` lists only the packages the run imports, with the Python version in its header.
- **Fixed seeds, single thread.** All estimators use `random_state=0` and `n_jobs=1`; no GPU is used.
- **Exported files.** The run writes the submission `evaluation/results/Cules of Duty.csv` and, in `runs/`, the run record as JSON (`Cules of Duty_config.json`, `Cules of Duty_final_model.json`) and as CSV (`Cules of Duty_run_summary.csv`: settings, library versions, pinned estimators, checksum); it prints the full path of each.
- **Data and caches stay out of git.** The match data and `cache/` are git-ignored; the repository holds code, the document and the submission file only.

The submitted file has MD5 `a62d4c09ce5107ad97c9be15b86e8a42` (printed by the run of the final notebook; the file is written to `evaluation/results/Cules of Duty.csv`). `reproduce.py` prints the MD5, and running it twice gives the same MD5 for reproducibility. Runtime on a laptop: 7 minutes.
