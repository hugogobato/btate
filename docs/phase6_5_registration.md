# Phase 6.5 Registration — Applied study on real perturbation data (sci-Plex)

**Status:** REGISTERED. Written and committed **before the first persistence
diagram is computed** on any real dataset. Pilot and smoke runs compute no
numbers used to choose any value in this document; the pilot is a
mechanics/timing check only and is explicitly non-interpretive.

**Registered date:** 2026-08-03 (sci-Plex 3 `srivatsan_2020_sciplex3.h5ad`,
`799317 x 110983`; sci-Plex 2 `srivatsan_2020_sciplex2.h5ad`, `24262 x 58347`;
downloaded to `data/` via `pertpy.data.srivatsan_2020_sciplex2/3`).

**Commit hash:** recorded at the bottom of this file at commit time (this
commit). The commit hash is what goes in the paper.

**Scope of this registration:** Tasks 6.5.1–6.5.4 of `Research_Plan.md`.
Tasks 6.5.5 (controls), 6.5.6 (causal layer report), 6.5.7 (depth audit),
6.5.8 (gates) are out of scope here; their inputs are pre-registered where
this document must not change after the fact (gates below).

**North star (Research_Plan.md):** the contribution is coverage, never
accuracy. Every reported number is a coverage rate (or an explicitly-labelled
interval-score / width / error diagnostic). Nothing in this document is chosen
by looking at evaluation coverage.

---

## 1. Unit of analysis (Task 6.5.1)

`i` = one independently treated cell population = one (plate, well) in sci-Plex 3
with a non-null `cell_line` annotation. A unit is the smallest population that
was independently cultured, treated and sequenced; cells within a unit are
never split across the calibration/evaluation boundary.

**Minimum-cell threshold (registered):** 60 recovered cells per unit, fixed
before looking at any diagram. This is a deliberate deviation from the plan's
proposal of 75 (Section 12, D1): the registered primary contrast's treated arm
consists of top-dose wells whose recovered cell counts are 13–117 (top doses
kill cells), and the plan's proposal would reduce the treated arm to two units.
The deviation is registered with the drop table below, before any diagram.

Drop table (from obs metadata only, 2026-08-03):

| Population | Units | Units >= 60 cells | Dropped |
|---|---|---|---|
| A549, all wells | 864 | 825 | 39 |
| K562, all wells | 768 | 731 | 37 |
| MCF7, all wells | 768 | 730 | 38 |
| A549 Alisertib top dose (10,000 nM) | 4 | 3 | 1 (13 cells) |
| A549 vehicle (control) | 24 | 24 | 0 |

## 2. Treatment and primary contrast (Task 6.5.1)

Prespecified before any diagram is computed; no compound or dose is selected
after seeing `psi`.

- **Primary binary contrast:** Alisertib (MLN8237), an Aurora-kinase-A
  inhibitor with established mitotic-arrest (cell-cycle-arrest) pharmacology,
  at its **top dose of 10,000 nM** versus vehicle (`perturbation == "control"`,
  `dose_value == 0`), **within A549**, pooling the 24 h and 72 h time points
  (time enters the covariate set, Section 4).
- **Rationale for pooling times:** at 24 h the registered top-dose treated arm
  has 1 eligible well (72 cells); pooling 24 h + 72 h gives 3 eligible treated
  wells (72, 90, 84 cells) against 24 eligible vehicle wells (194–300 cells),
  i.e. 27 evaluation units. A contrast with a 1-well treated arm is not
  estimable. Time is a design covariate, not a treatment level.
- **Evaluation units (pinned, well ids):** treated = `plate4_E9` (plate34,
  rep2, 72 cells), `plate2_E3` (plate49, rep1, 90 cells), `plate10_E9`
  (plate52, rep2, 84 cells); the 13-cell well `plate3_E3` (plate9, rep1) is
  excluded by the registered threshold. Vehicle = the 24 A549 `control` wells
  on plates 33–40 (16 wells, 24 h) and 49–52 (8 wells, 72 h).
- **Target estimand:** `psi_full(t)`, the AIPW contrast computed on the 27
  evaluation units **at full observed depth** (p = 1.0), with the estimator of
  Section 5. `psi_full` is a well-defined functional of the full-depth
  data-generating process; it is *not* a claim about noiseless biological
  truth (Research_Plan.md Task 6.5, "shallow-reference" caveat). It is computed
  once, before any replicate, and is fixed for all arms and all p.
- **Secondary contrast (registered, not computed in 6.5.1–6.5.4):** sci-Plex 2
  dose-response, seven dose levels {0, 0.1, 0.5, 1, 10, 50, 100} micromolar,
  four compounds {Dex, Nutlin, BMS, SAHA}, triplicate wells; `|psi|`
  monotone-in-dose check for compounds with known monotone pharmacology
  (Task 6.5.5 substrate).

## 3. Confounders (Task 6.5.1)

`X_i` = {time (24 h / 72 h indicator), log recovered cell count, plate
one-hot}. These are the only design variables that can induce finite-sample
imbalance. `e(X)` is the **known design value**: with randomization across
wells, `e(X) = n_treated / n_units = 3/27` for every unit, fixed, never
estimated. A finite-sample balance check on plate, time, cell count is
reported alongside every evaluation table. Propensity estimation, overlap
diagnostics and sensitivity analyses are out of scope (Task 6.5.6).

## 4. Outcome representation (Task 6.5.1 + 6.5.3)

`phi(D_i)` = the power-weighted silhouette (`r = 3`) of the H1 alpha-complex
diagram of unit `i`, evaluated on a **frozen** grid. The functional estimand is
kept; no collapse to a scalar.

**Frozen frame (registered).** (a) Per-cell normalisation: each cell is
normalised by its **own** total counts (size factor = cell total / median cell
total over calibration cells; median frozen at frame-fit time), then `log1p`.
(b) PCA to 50 components, **fit once** on a registered capped subsample of
50,000 calibration-set full-depth cells (seeded, stratified ~1 per 92
calibration cells), never refit afterwards; every other cloud is projected
through the stored loadings. Frame identity is asserted by a test (Section 9).
(c) **Filtration dimension `d_alpha = 3`:** the alpha complex (and the DTM
feature) are computed on the first 3 principal components. Deviation D2,
Section 12: the plan proposed PCA-to-50 as the filtration space; the alpha
complex is the Delaunay triangulation, which is computationally intractable in
50 ambient dimensions, and the loop structure of cycling cells concentrates in
the leading PCs. The full 50-dimensional frame is retained on disk for the
Task-6.5.7 depth audit and the explained-variance report.

**Frozen grid (registered rule).** `upper = round(1.5 * max finite H1 death, 3)`
over **calibration-set full-depth clouds only** (alpha grid); the DTM feature
grid uses the same rule on the DTM H1 deaths of a capped sample of 80
calibration full-depth clouds at `k = 15`. Resolution 96 (Phase-6 value). The
rule is recorded, not just the numbers; the resulting `grid_upper` values are
computed by `representation.py` and reported in its hash artifact.

**Scale (registered).** Every unit is subsampled to a fixed `n = 60` cells
(seeded, deterministic); the alternative "cardinality as bridge predictor" is
rejected. `n = 60` equals the registered threshold, so every included unit
contributes exactly 60 points, and 60 points keeps the top-dose clouds (72–90
recovered cells) analysable.

**Noise model that stays frozen:** thinning (Section 6) acts on raw counts
*before* any normalisation; the thinned profile is normalised by its own
totals exactly as a shallow-data analyst would.

## 5. Causal layer (registered, infrastructure only)

The AIPW machinery is inherited from `btate.benchmarks.frequentist`
(`aipw_effect`, Fourier-basis ridge outcome regression, `n_basis = 5`,
Gaussian-multiplier-bootstrap simultaneous band, `n_boot = 1000`,
`alpha = 0.05`) with **`cross_fit = False`** (registered: Task 6.5.6 says no
cross-fitting, and with 27 units 2-fold cross-fitting of the outcome model is
degenerate). `pi_hat` = the fixed design value 3/27 (per replicate: the
registered pool treated fraction). The Bayesian arms reuse
`btate.causal.propagation` (FGP, Phase-4 defaults; `n_inducing` clamps to n;
`n_causal_draws = 60`; `n_plugin_draws = 480`) and
`btate.benchmarks.measurement_error_uq.MeasurementBridge`
(`n_measurement_draws = 16`, `bridge_k_alpha = 10`, `bridge_k_dtm = 6`, ridge
grid `(1e-3, 1e-2, 1e-1, 1, 10)`, scale prior `1e-6`, calibration holdout
fraction 0.25 unit-disjoint). The causal layer is identical in every arm; only
the treatment of measurement error varies (Task 6.5.6).

## 6. Paired calibration resource (Task 6.5.2)

**Split.** Calibration and evaluation are disjoint at the level of
`(compound, plate)`. Registered rule: evaluation pairs are
`(Alisertib, plate9/34/49/52)` and `(control, plate33..40, plate49..52)`;
calibration = every cell-line-annotated unit with >= 60 cells whose
`(compound, plate)` is not an evaluation pair. This yields 4,614 calibration
units (1,686 A549, 1,420 K562, 1,508 MCF7; includes 79 control-arm wells and
21 Alisertib wells on non-evaluation plates), computed from obs metadata on
2026-08-03; the artifact reports the exact figure from the same rule at run
time. No evaluation unit ever enters the bridge. Calibration pools units
**across arms without the treatment label** (both drug-treated wells of other
compounds and control-arm wells from companion plates contribute paired
curves), so the map cannot encode the effect it is later used to estimate.
The A549 subset (1,686 units) is the in-cell-line sub-pool; the full pool is
registered because the thinning map is an assay-level property and the
Task-6.5.7 transport audit is pre-specified to quantify any cell-line shift.

**Thinning.** For every calibration cell and every registered retention
fraction `p in {1.0, 0.5, 0.25, 0.125}`: `U'_ig ~ Binom(U_ig, p)` with a
**fixed per-cell seed**, stored. The per-cell seed is a deterministic 32-bit
PRF of the cell barcode and `p` (registered function in
`btate.applied.paired_data`); the barcode list is stored so the map is
reproducible. **Primary p = 0.25** (interior of the ladder; 0.5 is a mild
perturbation, 0.125 is dropout-dominated; 0.25 mirrors the most severe
Phase-6 noise cell where the blind arm failed worst). The ladder is the
real-data analogue of the Phase-6 noise cells `{0, 0.125, 0.1875, 0.25}`:
`p = 1.0` is the control cell that must show no correction (null-thinning
control, Task 6.5.5).

**No-leak assertion (hard, in code):** no thinned profile is ever normalised
by a full-depth size factor; every profile (full or thinned) is normalised by
its own total counts. A test asserts that the normalisation of a thinned
profile depends only on the thinned counts.

**Bridge.** Fitted per `p` on the calibration pairs (observed = thinned,
clean = full-depth), ridge chosen by held-out predictive log-density on a
**unit-disjoint** holdout of the calibration pool (25%), then refit on the
full pool. `p = 1.0` bridges are fitted too (identity should hold; they are
the Task-6.5.5 null-thinning control artifact).

**BLAS:** every run pins BLAS threads to 1 (`phase6-blas-reproducibility`
finding; seeds alone are insufficient in this codebase).

**Corruption-geometry diagnostics (registered formulas, per p; never asserted,
reported):**
1. change in effective cloud cardinality after thinning (cells with total
   counts > 0 before vs after, on the same 60-cell subsample);
2. Hausdorff distance (in the PC1–3 frame) and the mean of the
   per-coordinate Wasserstein-1 distances (PC1–3) between the full and
   thinned clouds of the same unit;
3. H1 feature-count ratio (thinned / full) and the fraction of thinned
   features matched (within `eps = max(0.05 * grid_upper, Hausdorff)`, via
   the bottleneck correspondence) to a full feature; unmatched thinned
   features are "new" (clutter-like);
4. bottleneck distance between the full and thinned alpha diagrams vs the
   stability bound `d_B <= c * d_H` with registered constant `c = 2`;
   violation is the signature of clutter-like (non-jitter) corruption.

## 7. Evaluation protocol (Task 6.5.4)

**Seven arms** (Phase-6 table, reproduced): M1 oracle full-depth AIPW; M2
blind thinned AIPW; M3 Bayesian causal-only plugin; M4 bridge Bayesian
(nested); M5 bridge frequentist multiple-imputation AIPW; M4-DTM and M5-DTM
(DTM-augmented bridges; DTM mass per Section 8). All bands are simultaneous
(sup-t multiplier bootstrap for M1/M2/M5; posterior sup-norm-standardised for
M3/M4). `cross_fit = False` everywhere.

**Replicates.** R = 100 seeded subsamples without replacement of
`n_rep = 20` of the 27 evaluation units; a replicate must contain >= 1
treated and >= 1 control unit (else skipped; ~1.2% of draws), drawn in
seeded order until 100 valid replicates are recorded. The same unit subset is
used by every arm and every p within a replicate (paired design). Per-unit
curves are precomputed once at every p; a replicate is a table lookup.

**Headline number.** `cov_sim_full(p)` = simultaneous coverage of the fixed
`psi_full` curve by each arm's band at retention `p`, with exact
Clopper–Pearson 95% intervals, reported at the primary `p = 0.25` and at all
ladder values. Coverage is read **against the real-data M1 (full-depth) arm
as the empirical reference level, never against 0.95** (the Phase-6 oracle
itself covered 0.85–0.92 simultaneously; Task 6.5.4).

**Alongside coverage, in every table:** interval score (Gneiting–Raftery of
the simultaneous band against the fixed `psi_full`) and mean simultaneous
band width. Methods are never ranked by coverage alone (the corrected arms
over-cover at 0.98–1.00 in Phase 6).

**Held-out bridge report:** bridge diagnostics (selected ridge, selected k,
held-out predictive log-density, held-out clean-alpha RMSE vs naive, held-out
pointwise calibration) stratified by `p`, plus support-overlap diagnostics
on depth, cell count and PH-summary distributions between the calibration
pool and the evaluation pool.

## 8. DTM mass-parameter ablation (Task 6.5.4, required)

- Ladder: `k in {5, 10, 15, 25}` on `n = 60`-point clouds.
- **Selection rule (registered, calibration-only):** for each `p`, the pair
  `(k*, ridge*)` maximising held-out predictive log-density on the
  unit-disjoint calibration holdout (25%) is selected; the headline M4-DTM /
  M5-DTM arms use `(k*, ridge*)` refit on the full calibration pool. `k` is
  **not** inherited from the synthetic `k = 15`, which was tuned against
  interior clutter (Task 6.5.2's reason, second consequence).
- The **ablation table** reports `cov_sim_full`, interval score and width for
  M4-DTM / M5-DTM at every `k` in the ladder at the primary `p = 0.25`. The
  ablation is a sensitivity report, not a selection step: no post-hoc choice
  of `k` for the headline is made from evaluation results.

## 9. Frozen `representation.py` (Task 6.5.3)

Shipped as a single module `btate/applied/representation.py` whose source
hash (sha256 of the module text) is recorded in every artifact. Local runs and
Colab shards must verify the hash before running. A test asserts frame
identity: projecting the same cell twice through the stored loadings is
bit-identical, and full/thinned profiles of one unit are projected through the
same frozen frame (no refit).

## 10. Seeds and reproducibility

Registered seed namespace (32-bit ints, one stream per concern):
`frame = 65001`, `thinning = PRF(barcode, p)` (Section 6), `subsample =
65002`, `calibration_split = 65003`, `bridge_fit = 65004`, `ridge_k_selection
= 65005`, `rep_draws = 65006`, `measurement_posterior = 65007`, `causal =
65008`, `bootstrap = 65009`, `diagnostics = 65010`. BLAS threads pinned to 1
in every run. Raw frames are persisted per shard and combined
deterministically (`combine_shards`, dedup on `(rep, method, p, k)`).

## 11. Gates (registered, Task 6.5.8 — interpreted only, not run here)

- **Go** if: held-out bridge calibration passes (registered threshold: the
  DTM-augmented bridge's held-out clean-alpha RMSE below the naive
  observed-as-clean RMSE at every positive `p`); the negative control covers
  zero at its nominal rate (Task 6.5.5, out of scope here); and the coverage
  ordering `M2 ~ M3 << M4 ~ M5 ~ M4-DTM ~ M5-DTM` holds at `p = 0.25` with
  Clopper–Pearson intervals excluding the null ordering, read against the
  M1 reference level.
- **Conditional go** if the bridge calibrates and controls pass but blind
  `cov_sim_full >= 0.5` at `p = 0.25` (registered magnitude that counts as
  "small").
- **Falsified** if held-out bridge calibration fails, if the negative control
  does not cover zero, or if the correction is non-trivial at `p = 1.0`.

## 12. Registered deviations from the plan's proposals

1. **D1 — minimum-cell threshold 60 instead of proposed 75.** The registered
   primary contrast's treated arm is top-dose wells (13–117 cells); 75 would
   leave 2 treated units (72 h only) and a non-estimable 24 h arm. Drop tables
   in Section 1. Decision made from obs metadata only, before any diagram.
2. **D2 — filtration dimension 3, not "PCA to 50".** The alpha complex is a
   Delaunay triangulation; 50-dimensional alpha complexes are computationally
   intractable, and the loop structure of cycling cells concentrates in the
   leading PCs. The 50-dimensional frame is retained (Section 4) for the
   Task-6.5.7 audit.
3. **D3 — time pooled (24 h + 72 h) with a time covariate** for the primary
   contrast, for the sample-size reason in Section 2.
4. **D4 — `cross_fit = False`** in the AIPW outcome regression, per Task
   6.5.6's explicit "no cross-fitting" and the 27-unit sample.
5. **D5 — calibration spans all three cell lines** (4,614 units) rather than
   A549 only, so the pool includes both arms and the Task-6.5.7 transport
   question is measured, not assumed; the A549-only sub-pool is reported in
   the bridge-fit artifact as a diagnostic.

## 13. What is not done here

Task 6.5.5 (positive/negative controls and dose monotonicity), Task 6.5.6a
(confounded appendix), Task 6.5.7 (depth transport on Replogle), Task 6.5.8
(gate report). The pilot (sci-Plex 2) is a mechanics and timing smoke test
only: its numbers are not evidence and are not used to choose any value in
this document.

---

**Committed as** `docs/phase6_5_registration.md` in the published repo
(`dist/btate-github`). Commit hash (of the commit containing this exact file):

```
PENDING — filled at commit time
```
