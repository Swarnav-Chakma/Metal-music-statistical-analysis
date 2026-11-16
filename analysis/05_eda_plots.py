"""
05_eda_plots.py – EDA visualisations for the report.

Produces (in analysis/):
    fig_subgenre_counts.pdf      – recording counts per subgenre
    fig_dispersion_by_subgenre.pdf  – D_g with CI (requires H1 results)
    fig_bpm_sc_scatter.pdf       – BPM vs spectral centroid coloured by subgenre
    fig_minor_share.pdf          – minor share per subgenre (H3)
    fig_tempo_by_subgenre.pdf    – BPM distributions per subgenre
    fig_before_after_scale.pdf   – column std before vs after preprocessing

Run after 04_hypothesis_tests.py.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from statistical_core import ROOT, DATA

ANALYSIS = ROOT / "analysis"
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load data ────────────────────────────────────────────────────────────────
meta_va = pd.read_csv(ANALYSIS / "03_meta_val.csv")
meta_tr = pd.read_csv(ANALYSIS / "03_meta_train.csv")
X_raw_full = np.load(DATA / "X.npy").astype(np.float64)
meta_all = pd.read_csv(DATA / "metadata.csv")
fnames   = pd.read_csv(DATA / "feature_names.csv")
names_arr = fnames["name"].values

# Saved metadata rows are the authoritative retained ordering.
meta_raw_va = meta_va.copy(); meta_raw_tr = meta_tr.copy()
assert np.array_equal(meta_all.iloc[meta_va.row].recordingmbid, meta_va.recordingmbid)
X_raw_va = X_raw_full[meta_va.row.to_numpy()]
X_raw_tr = X_raw_full[meta_tr.row.to_numpy()]

bpm_idx = np.where(names_arr == "rhythm.bpm")[0][0]
sc_idx  = np.where(names_arr == "lowlevel.spectral_centroid.mean")[0][0]
ks_minor_idx = np.where(names_arr == "tonal.key_scale=minor")[0][0]

SUBGENRES = sorted(meta_raw_va["subgenre_short"].unique())

# ── Figure 1: Recording counts per subgenre ──────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.3))
counts_tr = meta_tr["subgenre_short"].value_counts().reindex(SUBGENRES, fill_value=0)
counts_va = meta_va["subgenre_short"].value_counts().reindex(SUBGENRES, fill_value=0)
x_pos = np.arange(len(SUBGENRES))
ax.barh(x_pos, counts_tr.values, color="#5b8db8", label="Train", height=0.45)
ax.barh(x_pos + 0.45, counts_va.values, color="#e07b39", label="Validation", height=0.45)
ax.set_yticks(x_pos + 0.225)
ax.set_yticklabels(SUBGENRES, fontsize=6.5)
ax.set_xlabel("Recordings")
ax.set_title("Recording counts per subgenre (train vs validation)")
ax.legend(loc="lower right")
ax.invert_yaxis()
fig.savefig(ANALYSIS / "fig_subgenre_counts.pdf")
plt.close(fig)
print("fig_subgenre_counts.pdf saved.")

# ── Figure 2: Dispersion D_g by subgenre ─────────────────────────────────────
if (ANALYSIS / "04_H1_results.csv").exists():
    h1 = pd.read_csv(ANALYSIS / "04_H1_results.csv").sort_values("D_g", ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    y_pos = np.arange(len(h1))
    ax.barh(y_pos, h1["D_g"].values, color="#5b8db8", height=0.6, zorder=2)
    ax.errorbar(h1["D_g"].values, y_pos,
                xerr=[h1["D_g"].values - h1["D_g_ci_lo"].values,
                      h1["D_g_ci_hi"].values - h1["D_g"].values],
                fmt="none", color="black", capsize=2.5, linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(h1["subgenre"].values, fontsize=6.5)
    ax.set_xlabel("Cluster-weighted mean distance from subgenre centroid ($D_g$)")
    ax.set_title("H1: Acoustic dispersion per subgenre (validation), 95% CI")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, zorder=0)
    fig.savefig(ANALYSIS / "fig_dispersion_by_subgenre.pdf")
    plt.close(fig)
    print("fig_dispersion_by_subgenre.pdf saved.")

# ── Figure 3: BPM vs spectral centroid scatter ───────────────────────────────
cmap = plt.get_cmap("tab20", len(SUBGENRES))
fig, ax = plt.subplots(figsize=(3.5, 3.5))
for i, sg in enumerate(SUBGENRES):
    mask = meta_raw_va["subgenre_short"] == sg
    ax.scatter(X_raw_va[mask, bpm_idx], X_raw_va[mask, sc_idx] / 1000,
               s=3, alpha=0.35, color=cmap(i), label=sg, rasterized=True)
ax.set_xlabel("Tempo (BPM)")
ax.set_ylabel("Spectral centroid mean (kHz)")
ax.set_title("H2: Tempo vs spectral brightness by subgenre (validation)")
ax.legend(loc="upper left", ncol=2, markerscale=2.5, fontsize=5.5,
          framealpha=0.7, handlelength=0.8)
fig.savefig(ANALYSIS / "fig_bpm_sc_scatter.pdf")
plt.close(fig)
print("fig_bpm_sc_scatter.pdf saved.")

# ── Figure 4: Minor share per subgenre ───────────────────────────────────────
if (ANALYSIS / "04_H3_results.csv").exists():
    h3 = pd.read_csv(ANALYSIS / "04_H3_results.csv").sort_values("minor_share", ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    y_pos = np.arange(len(h3))
    colors = ["#c94040" if v else "#5b8db8" for v in h3["beyond_SESOI"].values]
    ax.barh(y_pos, h3["minor_share"].values, color=colors, height=0.6, zorder=2)
    ax.errorbar(h3["minor_share"].values, y_pos,
                xerr=[h3["minor_share"].values - h3["ci_lo"].values,
                      h3["ci_hi"].values - h3["minor_share"].values],
                fmt="none", color="black", capsize=2.5, linewidth=0.8)
    overall_minor = float(h3["overall_minor"].iloc[0])
    baseline = ax.axvline(overall_minor, color="gray", linestyle="--", linewidth=1, label=f"Weighted mean {overall_minor:.2f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(h3["subgenre"].values, fontsize=6.5)
    ax.set_xlabel("Cluster-weighted estimated minor share")
    ax.set_title("H3: Estimated minor share (validation), 95% CI")
    ax.invert_yaxis()
    red_patch = mpatches.Patch(color="#c94040", label="Beyond SESOI (±10pp)")
    blue_patch = mpatches.Patch(color="#5b8db8", label="Within SESOI")
    ax.legend(handles=[red_patch, blue_patch, baseline], fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    fig.savefig(ANALYSIS / "fig_minor_share.pdf")
    plt.close(fig)
    print("fig_minor_share.pdf saved.")

# ── Figure 5: Tempo distributions per subgenre ───────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.3))
bpm_by_sg = {sg: X_raw_va[meta_raw_va["subgenre_short"] == sg, bpm_idx]
             for sg in SUBGENRES}
medians = [(sg, np.median(v)) for sg, v in bpm_by_sg.items()]
medians_sorted = sorted(medians, key=lambda x: x[1], reverse=True)
order = [sg for sg, _ in medians_sorted]
data_ordered = [bpm_by_sg[sg] for sg in order]
bp = ax.boxplot(data_ordered, vert=False, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.2),
                boxprops=dict(facecolor="#5b8db8", alpha=0.6),
                flierprops=dict(marker=".", markersize=1.5, alpha=0.3),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8))
ax.set_yticklabels(order, fontsize=6.5)
ax.set_xlabel("Tempo (BPM)")
ax.set_title("Tempo distributions per subgenre (validation, sorted by median)")
ax.grid(axis="x", alpha=0.3)
fig.savefig(ANALYSIS / "fig_tempo_by_subgenre.pdf")
plt.close(fig)
print("fig_tempo_by_subgenre.pdf saved.")

# ── Figure 6: Column std before vs after ─────────────────────────────────────
feat_summary = pd.read_csv(ANALYSIS / "01_feature_summary.csv")
# Before: std of original columns (non-constant, analysis-set cols)
selected = pd.read_csv(ANALYSIS / "03_columns_H1.csv").col_index.to_numpy()
std_before = X_raw_tr[:, selected].std(axis=0)
std_before = std_before[std_before > 0]
# After R1: load and compute column stds
X_va_r1 = np.load(ANALYSIS / "03_X_train_R1.npy").astype(np.float64)
std_after = X_va_r1.std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8))
axes[0].hist(np.log10(std_before + 1e-30), bins=50, color="#5b8db8", edgecolor="none")
axes[0].set_xlabel("log₁₀(raw SD)")
axes[0].set_ylabel("Count")
axes[0].set_title("Before")
axes[1].hist(std_after, bins=50, color="#e07b39", edgecolor="none")
axes[1].set_xlabel("R1 SD")
axes[1].set_title("After R1")
fig.suptitle("Training feature scales", y=1.01)
fig.tight_layout()
fig.savefig(ANALYSIS / "fig_before_after_scale.pdf")
plt.close(fig)
print("fig_before_after_scale.pdf saved.")

# ── Figure 7: Slopes per subgenre (H2) ───────────────────────────────────────
if (ANALYSIS / "04_H2_results.csv").exists():
    h2 = pd.read_csv(ANALYSIS / "04_H2_results.csv").sort_values("slope_beta_g", ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    y_pos = np.arange(len(h2))
    colors_h2 = ["#c94040" if v else "#5b8db8" for v in h2["beyond_SESOI"].values]
    ax.barh(y_pos, h2["slope_beta_g"].values, color=colors_h2, height=0.6, zorder=2)
    ax.errorbar(h2["slope_beta_g"].values, y_pos,
                xerr=[h2["slope_beta_g"].values - h2["ci_lo"].values,
                      h2["ci_hi"].values - h2["slope_beta_g"].values],
                fmt="none", color="black", capsize=2.5, linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(h2["subgenre"].values, fontsize=6.5)
    ax.set_xlabel(r"$\hat\beta_g$ (log centroid per log₂ BPM doubling)")
    ax.set_title("H2: Tempo–brightness slope per subgenre (validation), 95% CI")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, zorder=0)
    red_patch = mpatches.Patch(color="#c94040", label="Beyond SESOI (|β−β̄|>0.05)")
    blue_patch = mpatches.Patch(color="#5b8db8", label="Within SESOI")
    ax.legend(handles=[red_patch, blue_patch], fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.savefig(ANALYSIS / "fig_slopes_by_subgenre.pdf")
    plt.close(fig)
    print("fig_slopes_by_subgenre.pdf saved.")

print("\nAll EDA plots saved.")
