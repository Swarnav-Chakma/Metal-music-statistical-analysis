# Analysis plan 02: acoustic heterogeneity and feature relationships across metal subgenres

Status: **protocol, written before any hypothesis test was run.** No H1–H3 test statistic, p-value or confidence interval has been computed. Once approved, this plan is frozen. Any later change is logged in §11 with its date and reason, before the confirmatory run.

Inputs: `~/metal_data/features/` (X.npy SHA-256 `79a6605a…`, 5,874 × 2,675), `analysis/01_feature_summary.csv`, `analysis/01_distance_share_by_group.csv`. Design-time summaries used here are reproducible with `analysis/02_design_summaries.py` (outputs `02_design_summaries.json`, `02_clusters_by_subgenre.csv`).

---

## 1. Research objective and scope

Do the 17 Last.fm metal subgenres differ in **how spread out** their recordings are in acoustic feature space (H1), and in **how acoustic characteristics relate to each other** (H2: tempo and brightness), beyond differences in average feature values? H3 asks whether estimated tonality (major/minor) is associated with subgenre.

The plan does not cover retrieval, classification, clustering or the choice of a retrieval metric.

**What the data can and cannot show.** All quantities are computed from Essentia descriptors of commercial recordings, labelled by Last.fm crowd tags. A subgenre with larger acoustic dispersion has more varied *descriptors*. That variation can come from production era, mastering, recording quality, tag noise or the breadth of tag usage. It is not evidence of greater *artistic diversity*. Every association reported here is observational and is not evidence of causation.

---

## 2. Data, analysis set and the independent unit

### 2.1 Analysis set (fixed before testing)

| Step | Rule | Train | Validation |
|---|---|---|---|
| Start | Verified matrix | 3,108 | 2,766 |
| Exclude extractor build `2.1-dev` | 14 recordings, too few to estimate a version offset (heavy and industrial metal only) | −3 | −11 |
| Duplicate audio | Keep the lowest recording ID of each identical-checksum pair | −1 | −1 |
| **Analysis set** | | **3,104** | **2,754** |

The analysis script must assert these counts.

### 2.2 Exploratory vs confirmatory data

| Split | Role |
|---|---|
| **Training (3,104)** | All preprocessing parameters are fitted here: transforms, scaling, version offsets, family weights and thresholds. All three analyses are also run here first as an **exploratory pilot**, which debugs the code and the resampling. Pilot results are reported and labelled exploratory. They may not be used to change hypotheses, variables or tests. Any change prompted by the pilot must be logged in §11 *before* the validation run. |
| **Validation (2,754)** | **Confirmatory** tests are run exactly once, with every parameter frozen from training. |

Known limits to this separation:
- 360 artists have recordings in both splits (on different albums). Training-fitted preprocessing has therefore seen other recordings by some validation artists. Preprocessing only uses marginal, label-free statistics, apart from the version offsets (§3.3), so the leakage risk is small, but it is reported.
- **Disclosure:** while checking table sparsity for H3 during design, the observed **validation** major/minor counts by subgenre were printed once. That partially unblinds H3a's descriptive outcome. The summary script has since been restricted to margin-only expected counts. Because of this, H3a stays in the secondary family and its confirmatory status carries this disclosure. If the reviewer prefers, H3a can be downgraded to exploratory.

### 2.3 Is the track the independent unit? No.

Design-time check (`02_design_summaries.json`):

| Unit | Units with > 1 recording | Label-consistent | Units spanning > 1 subgenre | Recordings in inconsistent units |
|---|---|---|---|---|
| Album (release group) | 1,138 | 91.0% | 102 | 281 |
| Artist (first artist MBID) | 797 | 84.6% | 123 | 619 |
| **Cluster** (artist if known, else album) | 975 | 85.9% | 138 | 658 |

Two consequences:
1. **Tracks are not independent.** Tracks by one artist share production, line-up and mostly their label (P(same subgenre | same artist) = 0.95 in audit 01), so ordinary track-level permutation tests are **not** used.
2. **Clusters are not nested within subgenres.** 138 clusters contribute tracks to more than one subgenre, mainly neighbouring pairs such as alternative/nu metal and heavy/power metal. Permuting subgenre labels between whole clusters is therefore not exact. It is used only as a sensitivity analysis on the 2,166 label-consistent clusters.

**Primary independent unit: the cluster** = the recording's first artist MBID, or its album (release group) if the artist is unknown (15% of recordings). Artist is preferred over album because albums by the same artist are not independent. Unknown-artist recordings fall back to album clusters, which may understate dependence. A sensitivity analysis restricts to known-artist recordings.

**Primary resampling: the cluster bootstrap.** Whole clusters are resampled with replacement, keeping all of their tracks, including those in other subgenres. Resampling is stratified by each cluster's modal subgenre, with ties broken by the lowest subgenre name. This keeps within-cluster dependence, handles clusters that cross subgenres, and gives both p-values and confidence intervals.

**Effective sample size.** Validation clusters per subgenre range from 26 (folk) to 177 (black metal). Grindcore has 28 clusters, but one artist supplies 57 of its 158 tracks: with equal track weights its Kish effective number of clusters is **6.0**. **All primary estimates therefore weight clusters equally within each subgenre** (track weight = 1 / (tracks of that cluster in the subgenre × clusters in the subgenre)). Track-weighted estimates are a sensitivity analysis.

---

## 3. Feature representation (shared by H1)

Each step is fitted on training rows only and then applied unchanged to validation.

### 3.1 Column selection

| Step | Columns | Rationale |
|---|---|---|
| Verified matrix | 2,675 | |
| Drop constant (34), exact duplicates (330), near-constant non-categorical (48), `rhythm.beats_count` (1) | 2,262 | Audit 01 §4. The duplicates are the mirrored halves of the symmetric cov/icov matrices and BPM-peak mean = median = min = max. `beats_count` is a track-length proxy (ρ = 0.90), and length was deliberately excluded as file metadata. |
| Drop key/scale one-hots (`key_key`, `key_scale`, `chords_key`, `chords_scale`; 28) | **2,234 columns in 69 families** | They are nominal estimates: Euclidean distance between one-hot keys is arbitrary (C and C♯ would be as far apart as C and F♯). They are unreliable (§6.2) and are tested separately in H3. `key_strength`, the HPCP/THPCP and the chord histogram stay in. |

The resulting column list is written to `analysis/02_columns_H1.csv` by the analysis script, with an assertion of 2,234 columns.

### 3.2 Transformation

**Primary:** a per-column **rank-based inverse-normal transform (INT)** fitted on training rows. A value maps to Φ⁻¹ of its mid-rank empirical CDF, using Blom offsets. Validation values are mapped by linear interpolation of the training CDF and clipped to the training range, so every output is finite.

*Why:* column scales span 33 orders of magnitude and 769 columns have |skew| > 10. A single monotone, parameter-free rule avoids choosing a transform for each column by hand, and it stops a few heavy-tailed columns dominating the distance.

*Cost:* distances become rank-based, so tail differences are compressed. This is why a scale-based representation is a sensitivity analysis (§3.5).

### 3.3 Extractor-version adjustment

Audit 01 found that within subgenres, version explains > 10% of variance in 70 columns (max η² = 0.30; spectral roll-off, zero-crossing rate, high-frequency content, upper bands). The share of 2.1-beta1 recordings varies by subgenre (6–35%). Unadjusted, subgenres with more beta1 recordings would look more dispersed.

**Primary:** after the INT, each column is regressed on training rows as `x ~ subgenre + version`, giving 17 subgenre intercepts and a beta1 indicator. The fitted beta1 coefficient is subtracted from beta1 rows in both splits, which maps beta1 onto the beta2 reference. This removes **location** shifts only. Including subgenre in the regression stops the correction from absorbing real subgenre differences.

**Sensitivity:** (V1) no adjustment; (V2) beta2-only analysis, with each subgenre still required to keep ≥ 20 validation clusters; (V3) drop the version-sensitive columns (within-subgenre η² > 0.10 on training, 70 columns) without any adjustment.

### 3.4 Family-balanced weighting (primary)

After the INT and the version adjustment, each column is centred and scaled by its training mean and SD. Each column in family *f* is then multiplied by 1/√k_f, where k_f is the number of columns in the family. Families are the `family` field of `01_feature_summary.csv`: the descriptor path without statistic, index or level, for example `lowlevel.mfcc` or `lowlevel.spectral_centroid`.

**Effect:** each of the 69 families contributes the same expected squared distance (1 unit), whether it has 360 columns (ERB bands) or 1 (`rhythm.bpm`). Under plain column-wise scaling, six band families would take ~75% of the distance (audit 01 §2.5).

**Distance:** Euclidean distance on the weighted vectors.

*Limitation:* families still overlap in content. Bark, mel and ERB bands all describe the spectral envelope, so it carries 3/69 of the weight rather than 1/69. Representation R3 (§3.5) tests the effect of collapsing them.

### 3.5 Representations compared

| ID | Representation | Role |
|---|---|---|
| **R1** | INT → version adjustment → family-balanced weights → Euclidean | **Primary** |
| R2 | INT → version adjustment → equal column weights (conventional z-scored features) → Euclidean | Sensitivity: isolates the effect of family weighting |
| R3 | As R1, but Bark, mel and ERB bands merged into one "spectral envelope" family, and MFCC and GFCC merged into one "cepstral" family (66 families) | Sensitivity: cross-family redundancy |
| R4 | Robust scaling on the original values (training median and IQR, clipped to ±10) → version adjustment → equal column weights | Sensitivity: a conventional, non-rank scaling |

Raw Euclidean distance on the original values is **not used** (audit 01: one kurtosis family carries ~100% of raw variance).

---

## 4. H1 (primary): multivariate acoustic dispersion

### 4.1 Hypotheses

Let D_g be the cluster-weighted mean distance of subgenre *g*'s validation recordings from the subgenre centroid, in representation R1.
- **H1₀:** D_1 = D_2 = … = D_17.
- **H1₁:** at least one D_g differs.

### 4.2 How this differs from a test of centroids

PERMANOVA and MANOVA ask whether subgenre **centroids** (average acoustic profiles) differ: a test of *location*. PERMDISP asks whether recordings are **equally spread around their own centroid**: a test of *dispersion*. Two subgenres can share a centroid but differ in spread, or differ in centroid but share spread. A centroid test is also sensitive to dispersion differences when groups are unbalanced, which is one reason dispersion is examined directly. This plan tests dispersion only. Centroid differences are expected (they are what makes retrieval possible) and are not part of H1.

### 4.3 Statistic (PERMDISP-type, cluster-weighted)

For subgenre *g*, using the weights *w_i* from §2.3:
1. centroid c_g = Σ w_i x_i (cluster-weighted mean);
2. distance z_i = ‖x_i − c_g‖;
3. D_g = √(m_g / (m_g − 1)) · Σ w_i z_i, where m_g is the number of clusters in *g*. The factor corrects the small-sample bias of distances to an estimated centroid, as in Anderson's PERMDISP, with clusters as the effective units.

Omnibus statistic: the **Wald statistic** W = (A log d)ᵀ (A Σ Aᵀ)⁻¹ (A log d). Here d = (D_1 … D_17), A is the 16 × 17 contrast matrix of each log D_g against the mean of the log D_g, and Σ is the cluster-bootstrap covariance of log d. Logs stabilise the variance and make the contrasts ratios.

### 4.4 Null distribution and p-value

**Primary:** a **null-imposed cluster bootstrap.**
1. Rescale each subgenre's deviations to the common dispersion: x′_i = c_g + (x_i − c_g) · D̄ / D_g, where D̄ is the geometric mean of the D_g. This makes H1₀ exactly true while keeping each subgenre's centroid and shape.
2. Estimate Σ once, from 999 stratified cluster-bootstrap draws of the observed data. The same Σ is used for the observed W and every bootstrap W*, which keeps the cost bounded; the simulation check below confirms the resulting test is calibrated.
3. Draw B = 9,999 stratified cluster-bootstrap samples of x′. In each, recompute the centroids, the D_g and W* with that fixed Σ.
4. p = (1 + #{W* ≥ W}) / (B + 1).

Reported alongside: W compared with χ²(16). If the two disagree at the α = 0.05 boundary, the bootstrap p is the decision.

**Method check before use.** The null-imposed bootstrap is run on training data made null, by rescaling as in step 1, with 500 Monte Carlo repetitions. The empirical rejection rate at α = 0.05 must lie within 0.03–0.07. If it doesn't, the χ² reference with the cluster-bootstrap Σ is used and the change is logged.

**Sensitivity permutation test (P1):** the classic PERMDISP residual permutation at **cluster level**, on label-consistent clusters only. Each cluster's mean distance is centred on its subgenre mean, residuals are permuted between clusters, and the Welch F statistic is used (9,999 permutations).

### 4.5 Effect sizes and uncertainty

- **D_g** for each subgenre, with 95% cluster-bootstrap percentile intervals (B = 4,999).
- **Relative dispersion** r_g = D_g / D̄ with 95% interval. The range ratio max D_g / min D_g is reported with an interval.
- **Smallest effect of practical interest (SESOI): ±10% relative dispersion** (|log r_g| ≥ log 1.10). Fixed before testing. For scale, the plan also reports, from training data, the dispersion difference between beta1 and beta2 recordings within subgenres. That is the size of a known technical artefact, and subgenre effects smaller than it are interpreted with caution.
- **Exploratory decomposition:** between-cluster dispersion (distances of cluster centroids from c_g) versus within-cluster dispersion (distances of tracks from their cluster centroid). This shows whether a subgenre is spread out because its artists differ from each other or because individual artists are internally varied.

### 4.6 Post-hoc comparisons

Only if H1₀ is rejected: 17 contrasts, log r_g against zero, each with a cluster-bootstrap Wald test. **Holm correction** across the 17 at α = 0.05. All 136 pairwise ratios are reported as exploratory, with Holm-adjusted p-values and intervals, but no confirmatory claims.

### 4.7 Sensitivity analyses for H1

R2, R3, R4 (representations); V1, V2, V3 (version); track-weighted D_g; known-artist recordings only; P1 (cluster permutation). Sensitivity results are not used for decisions and are not multiplicity-corrected. They are judged by whether the **direction and rank order** of r_g and the omnibus conclusion hold. Agreement is summarised with Spearman's ρ between the r_g of R1 and of each sensitivity variant, plus the overlap of subgenres beyond the SESOI.

### 4.8 What counts as evidence

| Outcome | Interpretation |
|---|---|
| p < 0.05 **and** ≥ 1 subgenre whose 95% interval for r_g lies entirely outside [0.91, 1.10] **and** that subgenre's direction holds in R2–R4 and V1–V3 | **H1 supported:** subgenres differ in acoustic dispersion by a practically relevant amount. |
| p < 0.05 but every interval overlaps the SESOI band | Statistically detectable but **practically small**: not reported as a meaningful difference. |
| p < 0.05 but the direction reverses in a representation or version sensitivity analysis | **Fragile:** the result depends on representation or extractor handling; no substantive claim. |
| p ≥ 0.05 | **Not supported.** This is not evidence of equal dispersion; the intervals show which differences are excluded. |

---

## 5. H2 (secondary): tempo–brightness relationship by subgenre

### 5.1 Variables (exact fields, verified present)

| Role | Field | Transformation | Rationale |
|---|---|---|---|
| Outcome *y* | `lowlevel.spectral_centroid.mean` (Hz) | log_e | Ratio-scaled. Slopes read as proportional changes. The script asserts all values > 0. |
| Predictor *x* | `rhythm.bpm` | log₂, centred at the training median | The slope is the change in log centroid per doubling of tempo. Centring makes subgenre intercepts the brightness at typical tempo. |
| Covariate | extractor version (beta1 vs beta2) | indicator | Version explains 3.5% of within-subgenre centroid variance and 0.1% for BPM. |

Both raw fields are used here, not the INT features, so effects are on interpretable scales. Their skew is mild (0.21 and 0.25).

### 5.2 Hypotheses

Model (validation, cluster weights w_i = 1 / tracks of the cluster in that subgenre):

  y_i = α_g + β_g · x_i + γ · beta1_i + ε_i

- **H2₀:** β_1 = … = β_17 (the same tempo–brightness slope in every subgenre).
- **H2₁:** at least one β_g differs.

### 5.3 Test

- Weighted least squares, with a **CR2 cluster-robust covariance** clustered on the cluster defined in §2.3.
- Omnibus Wald test of the 16 slope contrasts.
- **p-value from a restricted wild cluster bootstrap:** Webb six-point weights, B = 9,999, with residuals from the model fitted under H2₀ (a common slope).
- The same simulation check as H1 is applied: 500 null datasets made by permuting x within subgenre on training data, with an acceptable rejection rate of 0.03–0.07.

### 5.4 Effect sizes and uncertainty

- β_g for each subgenre, with 95% wild-cluster-bootstrap percentile-t intervals, and the pooled slope.
- **Slope heterogeneity τ:** the cluster-weighted SD of the β_g, with a bootstrap interval.
- ΔR²: the weighted R² gain from adding the 16 interaction terms.
- **SESOI:** |β_g − β̄| ≥ 0.05, which is about a 5% difference in centroid per doubling of tempo.

### 5.5 Post-hoc

Only if H2₀ is rejected: 17 contrasts β_g − β̄ (the cluster-weighted mean of the other subgenres), each with the wild-cluster-bootstrap test, and **Holm** correction at α = 0.05.

### 5.6 Assumptions and sensitivity

- **Linearity in log–log within subgenres.** Checked on training data with binned residual plots, as exploratory work. If clearly violated, a sensitivity analysis replaces β_g with a per-subgenre **Spearman ρ** (cluster-weighted, with cluster-bootstrap intervals). This is also always reported as a robustness check.
- **Tempo-estimate errors (octave errors).** 1.8% of tracks have a second BPM peak at about double or half the first. Sensitivity T1 restricts to tracks with `rhythm.bpm_histogram_first_peak_weight.max` ≥ the training median (0.439), i.e. confident tempo estimates.
- **Version:** V-a adds a version × tempo interaction; V-b uses beta2 only.
- **Dependence:** M1 is a linear mixed model with a random intercept and slope per cluster (statsmodels MixedLM), with a likelihood-ratio test of the interaction. M2 uses track weights instead of cluster weights.

### 5.7 What counts as evidence

Supported if the omnibus p (after Holm with H3a, §7) < 0.05 **and** ≥ 1 subgenre's slope contrast interval excludes ±0.05 **and** the direction holds in T1, V-b and Spearman. Not supported if p ≥ 0.05. A significant interaction shows that tempo and brightness **co-vary differently** within subgenres in these recordings. It does not show that tempo causes brightness, or the reverse. Both depend on arrangement, production and mastering conventions, which differ by subgenre.

---

## 6. H3 (secondary): tonality and subgenre

### 6.1 Available tonal variables (verified present)

`tonal.key_key=*` (12 one-hot), `tonal.key_scale=*` (2), `tonal.chords_key=*` (12), `tonal.chords_scale=*` (2), `tonal.key_strength`. No other key estimator is in the data.

### 6.2 Reliability of the estimates (design-time check, all recordings)

- The two independent estimators (`key_*` and `chords_*`) agree on the **key** for 49.4% of tracks, on the **scale** for 60.8%, and on both for 39.2%.
- Agreement rises with `key_strength`: 21%, 33%, 47% and 58% across the training quartiles (cut points 0.574, 0.664, 0.737).
- Absolute key is confounded with **guitar tuning:** many metal bands tune down a semitone or more, which shifts the estimated key without any change in compositional mode.

Because of all this, **H3a uses scale (major/minor) only.** The 12-key analysis is exploratory.

### 6.3 H3a (confirmatory, secondary family; see the disclosure in §2.2)

- **Variable:** `tonal.key_scale` (minor = 1).
- **H3a₀:** P(minor) is the same in all 17 subgenres (scale is independent of subgenre).
- **H3a₁:** at least one subgenre differs.
- **Model:** cluster-weighted logistic regression, `minor ~ subgenre + version`, CR2 covariance.
- **Test:** omnibus Wald test of the 16 subgenre contrasts, with the p-value from a restricted wild cluster **score** bootstrap (B = 9,999).
- **Sparsity:** the validation 2 × 17 table has a minimum expected count of 20.3 and no cells below 5, so collapsing is not needed.

**Effect sizes:**
- Cluster-weighted minor share for each subgenre, with 95% cluster-bootstrap intervals.
- The odds ratio against all other subgenres.
- Cluster-weighted **Cramér's V** with a bootstrap interval.
- **SESOI:** a minor share differing by ≥ 10 percentage points from the cluster-weighted share of all other subgenres.

**Post-hoc:** only if rejected: 17 one-vs-rest contrasts, with **Holm** correction.

**Sensitivity:**
- K1: only tracks with `key_strength` ≥ the training median (0.664).
- K2: only tracks where `key_scale` = `chords_scale` (the two estimators agree).
- K3: `chords_scale` as the outcome.
- K4: beta2 only.
- K5: cluster-level permutation of subgenre labels on label-consistent clusters, using the χ² statistic.

**What counts as evidence:** supported if the Holm-adjusted omnibus p < 0.05 **and** ≥ 1 subgenre's interval lies beyond ±10 points **and** the direction holds in K1 and K2. Even then the conclusion concerns the **estimated** scale of recordings. Given ~61% agreement between estimators, it is weak evidence about the actual tonal practice of a subgenre.

### 6.4 H3b (exploratory only): 12 keys × subgenre

- **Why exploratory:** 16% of validation cells have expected counts below 5 (minimum 1.86), key agreement is only 49%, and tuning confounds absolute key.
- **What is reported:** cluster-weighted key distributions per subgenre, Cramér's V with a cluster-bootstrap interval, and a cluster-level permutation test (label-consistent clusters, χ² statistic, 9,999 permutations). The p-value is labelled nominal.
- **No sparse-cell pooling** is done for inference. Pooling keys would need a musical rule (for example by circle-of-fifths distance) chosen after seeing the data, which this plan avoids.

---

## 7. Significance, multiplicity and reporting rules

- **α = 0.05,** two-sided, fixed before testing.
- **Primary family:** H1 alone, at α = 0.05.
- **Secondary family:** H2 and H3a omnibus tests, **Holm** across the two at family-wise α = 0.05. They are tested whatever the outcome of H1, and their conclusions are labelled secondary.
- **Post-hoc contrasts:** only after a rejected omnibus test, with Holm within each hypothesis (17 contrasts).
- **Sensitivity analyses:** not corrected and not used for decisions. They are judged by direction and magnitude.
- **Exploratory analyses** (the training pilot, H3b, the dispersion decomposition, pairwise H1 ratios): no confirmatory claims; p-values labelled nominal.
- **Every reported test includes:** the statistic, its reference distribution (B or permutations), the p-value, the effect size, a 95% interval and the SESOI comparison. "Significant" is never written without the effect size.
- **Random seeds** are fixed and recorded (bootstrap seed 2025, permutation seed 2026). Resampling counts as stated.

---

## 8. Assumptions and how each is handled

| Assumption | Where | Handling |
|---|---|---|
| Independence between clusters | All | Clusters = artist, or album when the artist is unknown. Residual dependence via unknown artists → known-artist sensitivity. |
| Clusters nested in subgenres | Permutation tests | Violated for 138 clusters → bootstrap is primary; permutation only on label-consistent clusters. |
| Exchangeability under H₀ | H1 permutation | PERMDISP-style residual permutation only (location differences break raw label exchangeability); Welch F for unequal cluster variances. |
| Bootstrap validity with few effective clusters | H1–H3 | Cluster weighting (grindcore: 28 clusters, not Kish 6); simulation check of type-I error on training nulls before use. |
| Transform fitted on training generalises to validation | R1–R3 | Clipped to the training range; the number of clipped validation values per column is reported. |
| Version effect is a location shift | §3.3 | V2 (beta2 only) and V3 (drop sensitive columns) test it. |
| Log–log linearity | H2 | Exploratory training checks; Spearman sensitivity. |
| Tempo and key estimates are accurate | H2, H3 | T1, K1, K2 restrict to confident or agreeing estimates. |

---

## 9. Decision table

| Decision | Choice | Main rationale | Sensitivity |
|---|---|---|---|
| Confirmatory data | Validation split, run once | Preprocessing fitted on training; pilot on training is exploratory | — |
| Analysis set | Drop 2.1-dev (14) and one recording of each duplicate pair (2) → 3,104 / 2,754 | Too few dev recordings to adjust; identical audio is not independent | Include dev as beta2 |
| Independent unit | Cluster = artist, else album | 15% of artists and 9% of albums span subgenres; same-artist tracks share labels 95% of the time | Known-artist only; album clusters |
| Resampling | Stratified cluster bootstrap (null-imposed for H1; wild cluster for H2/H3) | Handles crossing clusters and unequal sizes; checked by simulation | Cluster permutation on label-consistent clusters |
| Weighting | Equal weight per cluster within subgenre | One artist = 36% of grindcore validation (Kish n = 6.0) | Track weights |
| H1 columns | 2,234 columns / 69 families | Audit drops plus nominal key one-hots | — |
| Transform | Training-fitted rank inverse-normal | 33 orders of magnitude of scale; 769 columns with \|skew\| > 10 | R4 robust scaling |
| Family balance | 1/√k_f per family | Stops six band families taking ~75% of distance | R2 equal weights; R3 merged envelope/cepstral |
| Version | Training-fitted location offset (subgenre-adjusted) | 70 columns with within-subgenre η² > 0.10 | V1 none; V2 beta2 only; V3 drop sensitive columns |
| H1 statistic | Cluster-weighted PERMDISP + Wald on log dispersions | Dispersion, not location; ratio-scale effects | Track-weighted; Welch-F permutation |
| H2 variables | log centroid ~ log₂ BPM × subgenre + version | Interpretable proportional slope | Spearman; confident-tempo subset; mixed model |
| H3 primary | Major/minor only | Key agreement 49%, tuning confound, 12 × 17 table sparse | K1–K5; H3b exploratory |
| α / multiplicity | 0.05; H1 alone; Holm for {H2, H3a}; Holm within post-hoc | Pre-specified hierarchy | — |
| SESOI | H1 ±10% dispersion; H2 ±0.05 slope; H3a ±10 points | Separates detectability from practical importance | Version-artefact size reported for context (H1) |

---

## 10. Planned outputs (next stage, after approval)

- `analysis/03_preprocess.py`: builds R1–R4 from training-fitted parameters and writes the column list and fitted parameters. No test statistics.
- `analysis/04_hypothesis_tests.py`: runs the simulation checks, then the training pilot (exploratory), then the confirmatory validation run, with seeds and B as above.
- `analysis/04_results.md`: results in the reporting format of §7, with exploratory and confirmatory results in separate sections.

## 11. Change log

| Date | Change | Reason | Made before validation results? |
|---|---|---|---|
| 2026-10-26 | Plan created | — | Yes (no test run) |

## 12. Final-validation amendment — 27 October 2026

The earlier plan above is retained as historical documentation, not an assertion that all planned analyses were executed. Earlier validation results had already been inspected. This amendment implements correctness repairs and the explicitly selected simpler final methods; it is not a new preregistration. Original scripts and outputs were preserved in an external timestamped backup before editing.

- Retain the original split and raw inputs. Verify 5,874 extracted rows, 14 dev-build exclusions and two duplicate-audio exclusions, yielding 5,858 rows (3,104 training / 2,754 validation).
- Use all valid credited artist UUIDs plus album release-group links to form stable connected-component dependence units. Malformed artist tokens do not link recordings. Components crossing subgenres remain intact. Split membership is never changed.
- Fit column selection on retained training data only and verify agreement with the original audit selection. Correct INT ties using average-rank Blom positions. Map validation using frozen unique training knots, linear interpolation of normal scores, and boundary clipping. Retain R1, R2 and R4 outputs, but do not claim sensitivity inference that was not run.
- H1: equal-cluster weighted mean Euclidean distance around each group's own centroid, without the unsupported small-sample multiplier. Use 1,999 successful whole-component bootstrap draws (seed 20261027) to estimate the covariance of log dispersions. Test 16 independent equality contrasts using an approximate chi-square Wald reference; obtain pointwise percentile CIs and a descriptive selected max/min-ratio interval. No null-imposed custom bootstrap or label permutation. No arbitrary ridge for a singular covariance.
- H2: WLS group-specific intercepts/slopes and extractor-version indicator; conventional CR1 covariance, complete-component scores, chi-square(16) slope-equality Wald. Report the equal-subgenre average slope and its normal 95% CI from the full slope covariance. Remove unverified custom wild bootstrap and CR2 claims.
- H3: explicitly exploratory saturated weighted linear probability model, CR1 covariance and chi-square(16) probability-equality Wald. No mixed-label permutation, version-adjustment claim or Cramer's V bootstrap CI. Report marginal estimated minor shares and unclipped pointwise normal intervals.
- Weights are 1/(n_cg m_g). Bootstrap occurrences of the same source cluster count separately; recalculate cell weights and centroids each draw. Whole-cluster multiplicity calculations are verified against literal expanded draws.
- H1 primary alpha=.05; Holm over available inferential H2 and H3 tests with sorted cumulative maximum. Full-rank finite contrast covariance, condition number <=1e10 and at least 20 clusters per subgenre are screening rules, not proofs of calibration. Use descriptive reporting if screens fail.
- Practical benchmarks are observed 10% relative dispersion, .05 slope departure and 10-percentage-point mode-share departure. They are analytical choices, not established perceptual thresholds. All hypotheses are omnibus; upper-tail statistics do not imply directional alternatives. No unadjusted post-hoc confirmatory claims.
- Acknowledge cross-split artists, uncertain mode estimates, asymptotic calibration and conditional-on-training uncertainty. Failed rejection does not establish equality; acoustic descriptor variation does not measure creativity.
