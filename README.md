# CSE558 Data Science Midterm — Acoustic Dispersion in Metal Subgenres

The authoritative report is **CSE558_Midterm_Report.pdf**, built from **CSE558_Midterm_Report.tex**. `report.pdf` and `report.tex` are synchronized copies. The final compiled report has **six pages**, uses the supplied course LaTeX template, and is within the eight-page limit.

Code, data-processing instructions and results are supplied locally. **There is no public GitHub repository yet. The public code/README link must be added before classroom submission.** The deadline is 27 October 2026, 11:59 pm.

## Data and scope

Inputs are in `~/metal_data/features/`: `X.npy`, `metadata.csv`, `feature_names.csv`, `feature_schema.json`, `extraction_report.json`, and `exclusions.csv`. No raw files or original split assignments are modified.

The local `~/metal_data/zenodo_manifest.json` identifies **Zenodo record 2553414**. Shard metadata identifies `acousticbrainz-mediaeval-features-train-89.tar.bz2` and `acousticbrainz-mediaeval-features-validation-01234567.tar.bz2`. This project uses their extracted subset, not the entire benchmark or all 30,388 candidate cohort records. No external source verification was used during final corrections.

| Population | Rows | Features | Instance–attributes |
|---|---:|---:|---:|
| Raw extracted | 5,874 | 2,675 | 15,712,950 |
| Retained base | 5,858 | 2,262 | 13,250,796 |
| H1 representation | 5,858 | 2,234 | 13,086,772 |

Exclude 14 development-build recordings (3 train / 11 validation) and two duplicate-audio recordings (1 / 1). Final split: **3,104 train / 2,754 validation**. The extraction-stage exclusion CSV is empty. There are no nonfinite raw feature values. The H1 features are acoustic descriptors, not columns created by one-hot encoding; 28 key/mode one-hots are excluded from H1 only.

## Final methods

All parameters and column-selection rules are fitted on retained training rows. Average-rank Blom normal scores handle ties; validation uses frozen unique training knots, interpolation and boundary clipping. R1 uses version correction, standardization and 69-family balancing. R2 and R4 matrices are retained but no sensitivity inference is claimed.

Dependence clusters are connected components linked by any valid credited artist UUID or album release-group ID. Multiple artists and transitive links are preserved. Malformed artist strings create no artist link. All tests use the same component IDs. Final counts: **2,222 retained components**, **1,566 training**, **1,004 validation**, **348 crossing splits**, and **79 validation components crossing subgenres**. At least 25 validation components support each subgenre. Largest component: 60 recordings, one artist, three albums; 57 of those recordings are validation. Component evidence is in `analysis/02_design_summaries.json`.

Each recording has weight `1/(n_cg*m_g)`: equal component weight within subgenre. H1 resamples complete validation components, carrying every represented subgenre along; repeated selections are separate occurrences. Bootstrap multiplicity calculations are tested against literal expanded resamples.

- **H1 primary:** weighted mean distance to each subgenre's own centroid in R1. Covariance of log dispersions from 1,999 whole-component bootstrap samples, seed 20261027; 16-contrast Wald test against approximate chi-square(16). No arbitrary ridge or old bias multiplier. Pointwise percentile CIs.
- **H2:** weighted group-specific intercept/slope regression of log spectral centroid on centered log2 tempo plus extractor-version indicator. Whole-component CR1 covariance, correction `M/(M-1)*(N-1)/(N-K)`, 16-contrast chi-square Wald. The pooled summary is the equal-subgenre average slope, with a normal 95% CI from the full covariance. It is not an imposed common slope.
- **H3 exploratory:** saturated weighted linear probability model for estimated minor mode; whole-component CR1 covariance and 16-contrast chi-square Wald. Marginal, not version-adjusted. Pointwise normal CIs are not clipped. No cluster-label permutation or Cramer's V bootstrap CI remains.

H1 alpha=.05. Holm applies to H2 and exploratory H3, using the sorted cumulative maximum. Statistical calibration is asymptotic; numerical checks are not proof of validity. The 10% dispersion, .05 slope and 10-percentage-point mode thresholds are project-defined practical benchmarks, not established psychoacoustic boundaries. Omnibus alternatives are non-directional, despite upper-tail test statistics.

## Regenerated results

| Test | Wald W (df=16) | Raw p | Holm p | Effect / interval |
|---|---:|---:|---:|---|
| H1 | 251.935 | 2.0748e-44 | Primary family | Max/min dispersion 1.35895, percentile CI [1.31095, 1.46808] |
| H2 | 20.869 | 0.183594 | 0.183594 | Equal-subgenre average slope 0.016369, normal CI [-0.033513, 0.066252] |
| H3 exploratory | 30.013 | 0.017937 | 0.035874 | Minor shares 0.417859–0.805333; group intervals in CSV |

H1 and exploratory H3 reject their equality nulls under the selected approximations; H2 does not. Failure to reject does not establish equal slopes. Descriptor dispersion is not artistic creativity. Selected extrema and counts beyond practical benchmarks are descriptive, not adjusted pairwise conclusions.

## Reproduce locally

The verified existing interpreter is `/opt/anaconda3/bin/python` (Python 3.13; NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3). Matplotlib creates figures; statsmodels independently validates CR1 estimates in the tests; pypdf validates PDF contents. The shell's plain `python3` points to a different environment lacking SciPy: do not use it for this pipeline. No installation is required in this environment.

Run from the project root:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/metal_matplotlib
/opt/anaconda3/bin/python analysis/03_preprocess.py
/opt/anaconda3/bin/python analysis/04_hypothesis_tests.py
/opt/anaconda3/bin/python analysis/02_design_summaries.py
/opt/anaconda3/bin/python analysis/05_eda_plots.py
/opt/anaconda3/bin/python analysis/test_final_pipeline.py > analysis/06_test_results.log 2>&1
tectonic --keep-logs CSE558_Midterm_Report.tex
cp CSE558_Midterm_Report.tex report.tex
cp CSE558_Midterm_Report.pdf report.pdf
/opt/anaconda3/bin/python analysis/06_verify_delivery.py
```

Do not rerun raw extraction or the initial audit. Report prose contains rounded final values; if methods or data change, reconcile it with newly generated CSV/JSON results before compiling. No script automatically invents report conclusions. For an output-only refresh with unchanged numerical inputs, `04_hypothesis_tests.py --reuse-bootstrap` verifies input/core hashes and reuses the saved 1,999 H1 draws; it recomputes regression estimates and output summaries. Changed numerical inputs require the normal full run.

## Files and verification

- `analysis/statistical_core.py`: shared clustering, transformation, weighting, CR1, Wald, Holm and bootstrap primitives.
- `analysis/03_preprocess.py`: training-fitted matrices/metadata/parameters; `03_quality_checks.json` records dimensions and input hashes.
- `analysis/04_hypothesis_tests.py`: writes `04_H1_results.csv`, `04_H2_results.csv`, `04_H3_results.csv`, `04_results.md`, `04_summary.json` and `04_inference_arrays.npz`.
- `04_summary.json`: numerical diagnostics, family membership, seeds and input/script hashes. `04_inference_arrays.npz` preserves the bootstrap draws and regression covariances for checking results without resampling.
- `analysis/02_design_summaries.py`: separates raw, retained, training and validation populations; saves component identifier evidence.
- `analysis/05_eda_plots.py`: seven `fig_*.pdf` files. The minor-share reference is the same equal-subgenre weighted mean used in H3. Plot titles contain no conflicting figure numbers.
- `analysis/test_final_pipeline.py`: tests ties/frozen maps, transitive clusters, row alignment, all six matrices, literal bootstrap equivalence, contrast covariance, independent statsmodels CR1 estimates, pooled CI, H3 shares, Holm, provenance and saved numerical results.
- `analysis/06_test_results.log` and `analysis/06_delivery_checks.json`: final verification evidence.
- `analysis/02_analysis_plan.md`: original plan preserved, with a dated final-validation amendment.
- `analysis/report_draft.tex`: historical draft, not a submission source.

## Limitations and history

This is an observational extracted subset with uncertain crowd tags and mode estimates (mode estimators agree on about 60.8%). Components approximate dependence from available IDs; missing links can leave residual dependence. There are artists/components in both splits. Inference conditions on training transformations and uses large-cluster approximations, with some subgenres having only 25 clusters. Bootstrap Monte Carlo error is finite; very small Wald tail probabilities should not be read as exact probabilities. No unused sensitivity analyses are claimed.

Earlier results had been viewed before these implementation corrections; the amendment does not pretend to restore untouched preregistration. H3 remains exploratory. The original Gemini-stage scripts, results, README and reports were preserved before editing in `/private/tmp/metal_final_backup_20261027_052225/project.tar.gz`, with SHA-256 manifest beside it. No raw dataset was copied into that backup.
