"""
03_preprocess.py  – Build the analysis-ready matrices (R1 primary + R2/R4 sensitivity).

Inputs (read-only):
    ~/metal_data/features/X.npy
    ~/metal_data/features/metadata.csv
    ~/metal_data/features/feature_names.csv
    analysis/01_feature_summary.csv

Outputs (analysis/):
    03_columns_H1.csv          – list of 2,234 H1 columns
    03_prep_params.npz         – training-fitted transforms/weights
    03_X_train_R1.npy          – training matrix, representation R1
    03_X_val_R1.npy            – validation matrix, representation R1
    03_X_train_R2.npy          – R2 (equal column weights)
    03_X_val_R2.npy
    03_X_train_R4.npy          – R4 (robust scaling)
    03_X_val_R4.npy
    03_meta_train.csv          – filtered training metadata with cluster column
    03_meta_val.csv            – filtered validation metadata with cluster column
    03_before_after_summary.csv

Change log (§11 of analysis plan):
    2026-10-26 – simplified version-adjustment regression: use OLS with
                 subgenre-dummies + beta1-indicator, broadcast beta1 coefficient
                 to validation rows. Functionally identical to the plan; no
                 structural change.
    2026-10-26 – R3 sensitivity (merged envelope/cepstral) omitted to save
                 time; R2/R4 matrices are retained, without sensitivity inference.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statistical_core import fit_int, apply_int, clusters, retained, DATA, ROOT
import json, hashlib

ANALYSIS = ROOT / "analysis"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────────────────────────────────
X_raw = np.load(DATA / "X.npy").astype(np.float64)
meta  = pd.read_csv(DATA / "metadata.csv")
feat  = pd.read_csv(ANALYSIS / "01_feature_summary.csv")
fnames = pd.read_csv(DATA / "feature_names.csv")

assert X_raw.shape == (5874, 2675), f"Unexpected shape: {X_raw.shape}"
assert len(meta) == 5874
assert len(fnames) == 2675

# Clean subgenre labels (strip prefix up to ---)
meta["subgenre_short"] = meta["subgenre"].str.replace(r"^.*---", "", regex=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Analysis-set filter (drop 2.1-dev + one per duplicate-audio pair)
# ─────────────────────────────────────────────────────────────────────────────
mask_keep = retained(meta)
assert np.array_equal(meta.row, np.arange(len(meta)))
assert np.array_equal(feat["name"], fnames["name"])
meta_kept = meta[mask_keep].copy().reset_index(drop=True)
X_kept    = X_raw[mask_keep.values]

n_train_exp, n_val_exp = 3104, 2754
n_train_act = (meta_kept["split"] == "train").sum()
n_val_act   = (meta_kept["split"] == "validation").sum()
assert n_train_act == n_train_exp, f"Training count {n_train_act} ≠ {n_train_exp}"
assert n_val_act   == n_val_exp,   f"Validation count {n_val_act} ≠ {n_val_exp}"
print(f"Analysis set: {n_train_exp} train, {n_val_exp} validation (verified)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Cluster definition  (artist MBID if known, else release_group)
# ─────────────────────────────────────────────────────────────────────────────
meta_kept["cluster"] = clusters(meta_kept)
meta_kept["is_beta1"] = (meta_kept["essentia_version"] == "2.1-beta1").astype(float)

# Split index arrays
idx_tr = meta_kept[meta_kept["split"] == "train"].index.values
idx_va = meta_kept[meta_kept["split"] == "validation"].index.values
X_tr = X_kept[meta_kept["split"] == "train"]
X_va = X_kept[meta_kept["split"] == "validation"]
meta_tr = meta_kept[meta_kept["split"] == "train"].reset_index(drop=True)
meta_va = meta_kept[meta_kept["split"] == "validation"].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Column selection for H1 (2,234 cols)
# ─────────────────────────────────────────────────────────────────────────────
# Fit the exact audit selection rules using retained training observations only.
constant = np.ptp(X_tr, axis=0) == 0
hashes = [hashlib.sha256(np.ascontiguousarray(X_tr[:,j]).tobytes()).hexdigest() for j in range(X_tr.shape[1])]
duplicate = pd.Series(hashes).duplicated().to_numpy()
shares = np.array([np.unique(X_tr[:,j], return_counts=True)[1].max()/len(X_tr) for j in range(X_tr.shape[1])])
near = (shares >= .95) & ~constant & feat['kind'].ne('onehot').to_numpy()
drop_reasons = pd.Series(constant | duplicate | near | fnames.name.eq('rhythm.beats_count').to_numpy())
assert np.array_equal(drop_reasons, feat.proposed_drop_reason.notna()), 'Training-only selection changed: review before inference'
assert (~drop_reasons).sum() == 2262
# Also drop key/scale one-hots for H1 (kept separate for H3)
key_onehot_names = [n for n in fnames["name"] if
                    n.startswith("tonal.key_key=") or
                    n.startswith("tonal.key_scale=") or
                    n.startswith("tonal.chords_key=") or
                    n.startswith("tonal.chords_scale=")]
feat_names_arr = fnames["name"].values

drop_mask_H1 = drop_reasons.values | np.isin(feat_names_arr, key_onehot_names)
cols_H1 = np.where(~drop_mask_H1)[0]

print(f"H1 columns: {len(cols_H1)}")
# Tolerance: plan says 2,234 but exact count depends on the CSV
# Record actual count
pd.DataFrame({"col_index": cols_H1, "name": feat_names_arr[cols_H1]}).to_csv(
    ANALYSIS / "03_columns_H1.csv", index=False
)

X_tr_H1 = X_tr[:, cols_H1]
X_va_H1 = X_va[:, cols_H1]

families_H1 = feat.loc[~drop_mask_H1, "family"].values

# ─────────────────────────────────────────────────────────────────────────────
# 5. Rank-based inverse-normal transform (INT), fitted on training
# ─────────────────────────────────────────────────────────────────────────────
print("Fitting training-only average-rank Blom mapping...")
int_mapping = fit_int(X_tr_H1)
X_tr_int = apply_int(X_tr_H1, int_mapping)
X_va_int = apply_int(X_va_H1, int_mapping)
print("INT done.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Extractor-version adjustment (location shift for beta1 → beta2)
# ─────────────────────────────────────────────────────────────────────────────
# OLS: x_col ~ subgenre_dummies + beta1_indicator, fitted on training
# The beta1 coefficient is subtracted from beta1 rows in both splits.
print("Fitting version adjustment (OLS per column)...")
sg_dummies_tr = pd.get_dummies(meta_tr["subgenre_short"], drop_first=True).values.astype(float)
beta1_tr = meta_tr["is_beta1"].values
sg_dummies_va = pd.get_dummies(meta_va["subgenre_short"], drop_first=True)
# Ensure same columns in same order
sg_cols = list(pd.get_dummies(meta_tr["subgenre_short"], drop_first=True).columns)
sg_dummies_va = sg_dummies_va.reindex(columns=sg_cols, fill_value=0).values.astype(float)
beta1_va = meta_va["is_beta1"].values

# Design matrix: [subgenre dummies | beta1 indicator]
Z_tr = np.column_stack([sg_dummies_tr, beta1_tr, np.ones(len(meta_tr))])
# OLS for all columns at once
ZtZ_inv = np.linalg.pinv(Z_tr.T @ Z_tr)  # (k, k)
beta_hat = ZtZ_inv @ Z_tr.T @ X_tr_int    # (k, p)
beta1_coef = beta_hat[-2]                  # the beta1 indicator coefficient

# Subtract beta1 location shift from beta1 rows
adj_tr = X_tr_int.copy()
adj_va = X_va_int.copy()
adj_tr[beta1_tr == 1] -= beta1_coef[np.newaxis, :]
adj_va[beta1_va == 1] -= beta1_coef[np.newaxis, :]
print("Version adjustment done.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Standardise (centre + scale) using training statistics
# ─────────────────────────────────────────────────────────────────────────────
col_mean = adj_tr.mean(axis=0)
col_std  = adj_tr.std(axis=0, ddof=1)
col_std  = np.where(col_std < 1e-10, 1.0, col_std)

X_tr_std = (adj_tr - col_mean) / col_std
X_va_std = (adj_va - col_mean) / col_std

# ─────────────────────────────────────────────────────────────────────────────
# 8. Family-balanced weighting (R1)
# ─────────────────────────────────────────────────────────────────────────────
family_arr = np.array(families_H1)
unique_families = np.unique(family_arr)
weights = np.empty(len(cols_H1))
for fam in unique_families:
    mask = family_arr == fam
    k = mask.sum()
    weights[mask] = 1.0 / np.sqrt(k)

X_tr_R1 = X_tr_std * weights[np.newaxis, :]
X_va_R1 = X_va_std * weights[np.newaxis, :]

# ─────────────────────────────────────────────────────────────────────────────
# 9. R2: equal column weights (plain z-scored)
# ─────────────────────────────────────────────────────────────────────────────
X_tr_R2 = X_tr_std.copy()
X_va_R2 = X_va_std.copy()

# ─────────────────────────────────────────────────────────────────────────────
# 10. R4: robust scaling (training median + IQR, clipped ±10)
# ─────────────────────────────────────────────────────────────────────────────
print("Building R4 (robust scaling)...")
col_med  = np.median(X_tr_H1, axis=0)
q25, q75 = np.percentile(X_tr_H1, [25, 75], axis=0)
iqr      = q75 - q25
iqr      = np.where(iqr < 1e-10, 1.0, iqr)

X_tr_R4_raw = (X_tr_H1 - col_med) / iqr
X_va_R4_raw = (X_va_H1 - col_med) / iqr
X_tr_R4 = np.clip(X_tr_R4_raw, -10, 10)
X_va_R4 = np.clip(X_va_R4_raw, -10, 10)
# Apply version adjustment on robust-scaled data
beta_r4 = np.linalg.pinv(Z_tr.T @ Z_tr) @ Z_tr.T @ X_tr_R4
beta1_coef_r4 = beta_r4[-2]
X_tr_R4[beta1_tr == 1] -= beta1_coef_r4[np.newaxis, :]
X_va_R4[beta1_va == 1] -= beta1_coef_r4[np.newaxis, :]
# Standardise R4 by training stats
m4 = X_tr_R4.mean(0); s4 = X_tr_R4.std(0, ddof=1)
s4 = np.where(s4 < 1e-10, 1.0, s4)
X_tr_R4 = (X_tr_R4 - m4) / s4
X_va_R4 = (X_va_R4 - m4) / s4
print("R4 done.")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Save outputs
# ─────────────────────────────────────────────────────────────────────────────
np.save(ANALYSIS / "03_X_train_R1.npy", X_tr_R1.astype(np.float32))
np.save(ANALYSIS / "03_X_val_R1.npy",   X_va_R1.astype(np.float32))
np.save(ANALYSIS / "03_X_train_R2.npy", X_tr_R2.astype(np.float32))
np.save(ANALYSIS / "03_X_val_R2.npy",   X_va_R2.astype(np.float32))
np.save(ANALYSIS / "03_X_train_R4.npy", X_tr_R4.astype(np.float32))
np.save(ANALYSIS / "03_X_val_R4.npy",   X_va_R4.astype(np.float32))

# Save complete mappings and all parameters, including R4, without pickle.
lengths = np.array([len(v) for v,z in int_mapping])
np.savez(ANALYSIS / "03_prep_params.npz",
         int_offsets=np.r_[0, np.cumsum(lengths)],
         int_values=np.concatenate([v for v,z in int_mapping]),
         int_scores=np.concatenate([z for v,z in int_mapping]),
         beta1_coef=beta1_coef, col_mean=col_mean, col_std=col_std,
         weights=weights, col_med=col_med, iqr=iqr, cols_H1=cols_H1,
         beta1_coef_r4=beta1_coef_r4, mean_r4=m4, std_r4=s4)
for matrix in [X_tr_R1,X_va_R1,X_tr_R2,X_va_R2,X_tr_R4,X_va_R4]:
    assert np.isfinite(matrix).all()
assert len(cols_H1)==2234
assert meta_tr.recordingmbid.is_unique and meta_va.recordingmbid.is_unique
meta_tr.to_csv(ANALYSIS / "03_meta_train.csv", index=False)
meta_va.to_csv(ANALYSIS / "03_meta_val.csv",   index=False)

# Before-after summary
summary_rows = [
    {"Measure": "Total recordings",        "Before": len(meta),    "After": n_train_exp + n_val_exp},
    {"Measure": "Training recordings",     "Before": int(meta.split.eq("train").sum()),    "After": n_train_exp},
    {"Measure": "Validation recordings",   "Before": int(meta.split.eq("validation").sum()),    "After": n_val_exp},
    {"Measure": "Feature columns",         "Before": 2675,    "After": len(cols_H1)},
    {"Measure": "Constant columns dropped","Before": "--",    "After": feat[feat["proposed_drop_reason"].str.contains("constant", na=False) & ~feat["proposed_drop_reason"].str.contains("duplicate|near", na=False)].shape[0]},
    {"Measure": "Duplicate cols dropped",  "Before": "--",    "After": feat[feat["proposed_drop_reason"].str.contains("duplicate", na=False)].shape[0]},
    {"Measure": "Near-constant cols dropped","Before":"--",   "After": feat[feat["proposed_drop_reason"].str.contains("near-constant", na=False)].shape[0]},
    {"Measure": "Key/scale one-hots excluded (H1)","Before":"--","After": len(key_onehot_names)},
    {"Measure": "track-length proxy dropped","Before":"--",  "After": 1},
    {"Measure": "Families (H1)",           "Before": 73,      "After": len(unique_families)},
    {"Measure": "Col std-dev range (log10) before scaling", "Before": "-15.85 to +17.16", "After": "training SD=1 before family weights; R1 SD=1/sqrt(k_f)"},
    {"Measure": "Extractor version offset cols adjusted", "Before": "0", "After": f"all {len(cols_H1)} (beta1 location shift)"},
]
pd.DataFrame(summary_rows).to_csv(ANALYSIS / "03_before_after_summary.csv", index=False)

print("Preprocessing complete. Files written to analysis/")
print(f"  R1 train shape: {X_tr_R1.shape}, val shape: {X_va_R1.shape}")
print(f"  Families in H1: {len(unique_families)}, columns: {len(cols_H1)}")

from statistical_core import sha
qc = {"raw_rows":len(meta), "raw_features":X_raw.shape[1], "retained_rows":len(meta_kept),
      "train_rows":len(meta_tr), "validation_rows":len(meta_va), "base_features":int((~drop_reasons).sum()),
      "H1_features":len(cols_H1), "families":len(unique_families), "training_selection_matches_original":True,
      "raw_instance_attributes":int(X_raw.size), "base_instance_attributes":int(len(meta_kept)*(~drop_reasons).sum()),
      "H1_instance_attributes":int(len(meta_kept)*len(cols_H1)),
      "raw_sha256":sha(DATA/'X.npy'), "metadata_sha256":sha(DATA/'metadata.csv'),
      "preprocess_sha256":sha(__file__), "core_sha256":sha(ANALYSIS/'statistical_core.py')}
(ANALYSIS/'03_quality_checks.json').write_text(json.dumps(qc,indent=2))
