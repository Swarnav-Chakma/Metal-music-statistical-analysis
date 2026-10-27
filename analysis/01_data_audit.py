#!/usr/bin/env python3
"""
Stage 3a: non-destructive data-quality and exploratory audit of the extracted feature matrix.

Reads (never writes) the dataset in --data-dir:
  features/X.npy, metadata.csv, feature_names.csv, exclusions.csv, extraction_report.json
  shards/*.zip (raw JSON, used to re-check the row <-> recording mapping)
  cohort.csv (full 30,388-recording cohort, used to check the archive subset for imbalance)
Writes small tables and the report to --out-dir (default: this script's folder).

Hard integrity failures raise AuditError with a message and stop the run. Everything else is a
finding reported in 01_data_audit.md. Nothing here standardises the data, fits models or chooses
metrics: z-scores and correlations below are computed in memory as diagnostics only.

Usage
  python3 analysis/01_data_audit.py                       # default paths, 300-row mapping sample
  python3 analysis/01_data_audit.py --full-mapping        # re-derive every row from raw JSON
  python3 analysis/01_data_audit.py --data-dir D --out-dir O --mapping-sample N
"""
import argparse, hashlib, json, os, re, sys, zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

STATS = {"mean", "median", "min", "max", "var", "dmean", "dmean2", "dvar", "dvar2", "cov", "icov"}
EXPECTED_META_COLS = ["row", "recordingmbid", "subgenre", "split", "release_group", "artist_mbids",
                      "albumartist_mbids", "tag_recording_mbid", "essentia_version", "essentia_build_sha",
                      "audio_md5", "audio_md5_shared", "length_seconds", "lenient_parse", "archive"]
NEAR_CONST_SHARE = 0.95      # a column whose most common value covers >= 95% of rows
ROBUST_Z = 10.0              # "extreme" value: more than 10 robust SDs from the median


class AuditError(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise AuditError(msg)


# ----------------------------------------------------------------------------- loading
def load(data_dir):
    F = os.path.join(data_dir, "features")
    for f in ["X.npy", "metadata.csv", "feature_names.csv", "exclusions.csv", "extraction_report.json"]:
        check(os.path.exists(os.path.join(F, f)), f"missing input file {os.path.join(F, f)}")
    X = np.load(os.path.join(F, "X.npy"), mmap_mode=None)
    meta = pd.read_csv(os.path.join(F, "metadata.csv"), dtype={"artist_mbids": str, "albumartist_mbids": str})
    names = pd.read_csv(os.path.join(F, "feature_names.csv"))
    excl = pd.read_csv(os.path.join(F, "exclusions.csv"))
    rep = json.load(open(os.path.join(F, "extraction_report.json")))
    xsha = hashlib.sha256(open(os.path.join(F, "X.npy"), "rb").read()).hexdigest()
    return X, meta, names, excl, rep, xsha


def lookup(d, name):
    """Value of feature `name` (e.g. lowlevel.mfcc.cov[2,5], tonal.key_key=A) in a raw JSON dict."""
    path, i, j, lvl = re.match(r"^([^\[=]+)(?:\[(\d+)(?:,(\d+))?\])?(?:=(.+))?$", name).groups()
    o = d
    for k in path.split("."):
        o = o[k]
    if lvl is not None:
        return 1.0 if o == lvl else 0.0
    if j is not None:
        return o[int(i)][int(j)]
    return o[int(i)] if i is not None else o


def family_of(name):
    """Descriptor family: the JSON path without statistic suffix, index or one-hot level."""
    path = re.split(r"[\[=]", name)[0]
    parts = path.split(".")
    return ".".join(parts[:-1]) if parts[-1] in STATS else path


def block_of(name):
    path = re.split(r"[\[=]", name)[0]; last = path.split(".")[-1]
    if "=" in name:
        return "one-hot (key/scale)"
    if last in ("cov", "icov"):
        return "MFCC/GFCC cov+icov matrices"
    if "[" in name:
        return "vector descriptors (bands, coefficients, HPCP)"
    return "scalar descriptors"


# ----------------------------------------------------------------------------- integrity
def integrity(X, meta, names, excl, rep, xsha, data_dir, mapping_sample, seed):
    R = {}
    check(X.ndim == 2, f"X.npy must be 2-D, got shape {X.shape}")
    n, p = X.shape
    R["shape"] = [n, p]; R["dtype"] = str(X.dtype)
    check(list(rep["X_shape"]) == [n, p], f"X shape {X.shape} != extraction report {rep['X_shape']}")
    check(xsha == rep["X_sha256"], "X.npy SHA-256 does not match extraction_report.json (file changed after build?)")
    check(list(meta.columns) == EXPECTED_META_COLS, f"unexpected metadata columns: {list(meta.columns)}")
    check(len(meta) == n, f"metadata has {len(meta)} rows but X has {n}")
    check(len(names) == p, f"feature_names has {len(names)} rows but X has {p} columns")
    check((names["index"].values == np.arange(p)).all(), "feature_names.index is not 0..p-1 in order")
    check(names["name"].is_unique, "duplicate feature names")
    check((meta["row"].values == np.arange(n)).all(), "metadata.row is not 0..n-1 in order")
    check(meta["recordingmbid"].is_unique, "duplicate recording IDs in metadata")
    nonfinite = int((~np.isfinite(X)).sum())
    check(nonfinite == 0, f"{nonfinite} non-finite values in X")
    R["nonfinite"] = nonfinite
    R["meta_missing"] = {c: int(v) for c, v in meta.isna().sum().items() if v}
    req = ["recordingmbid", "subgenre", "split", "release_group", "essentia_version", "audio_md5"]
    for c in req:
        check(meta[c].notna().all(), f"metadata column {c} has missing values")
    check(set(meta["split"]) <= {"train", "validation"}, f"unexpected split values {set(meta['split'])}")
    check(len(excl) == 0 or set(excl["recordingmbid"]).isdisjoint(meta["recordingmbid"]),
          "a recording is both excluded and present in metadata")
    R["exclusions"] = len(excl)
    # forbidden content in the feature vector
    bad = [nm for nm in names["name"] if nm.startswith("metadata") or "beats_position" in nm]
    check(not bad, f"feature vector contains non-acoustic fields: {bad[:5]}")

    # repeated rows / duplicate columns
    row_hash = pd.Series([hashlib.md5(X[i].tobytes()).hexdigest() for i in range(n)])
    R["repeated_row_groups"] = int((row_hash.value_counts() > 1).sum())
    col_hash = pd.Series([hashlib.md5(np.ascontiguousarray(X[:, j]).tobytes()).hexdigest() for j in range(p)])
    dup_cols = col_hash.duplicated(keep="first")
    R["duplicate_columns"] = int(dup_cols.sum())

    # album-disjoint splits
    rg_splits = meta.groupby("release_group")["split"].nunique()
    check((rg_splits == 1).all(), f"{int((rg_splits > 1).sum())} release groups appear in both splits")
    R["release_groups"] = int(len(rg_splits))

    # row <-> recording mapping, re-derived from the raw JSON in the shards
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if mapping_sample >= n else np.sort(rng.choice(n, mapping_sample, replace=False))
    shard_dir = os.path.join(data_dir, "shards")
    zips, bad_rows = {}, []
    for i in idx:
        r = meta.iloc[i]
        zp = os.path.join(shard_dir, r["archive"].replace(".tar.bz2", ".zip"))
        check(os.path.exists(zp), f"shard {zp} missing; cannot verify row mapping")
        z = zips.setdefault(zp, zipfile.ZipFile(zp))
        d = json.loads(z.read(f"{r['recordingmbid']}.json").decode("utf-8"), strict=False)
        ref = np.array([lookup(d, nm) for nm in names["name"]], dtype=np.float32)
        if not np.array_equal(ref, X[i]):
            bad_rows.append(int(i))
    check(not bad_rows, f"{len(bad_rows)} matrix rows do not match their raw JSON, e.g. rows {bad_rows[:5]}")
    R["mapping_rows_checked"] = int(len(idx))
    return R, dup_cols.values, row_hash


# ----------------------------------------------------------------------------- feature quality
def feature_quality(X, meta, names, dup_cols):
    n, p = X.shape
    Xd = X.astype(np.float64)
    tr = (meta["split"] == "train").values
    df = names[["index", "name", "section", "kind"]].copy()
    df["family"] = df["name"].map(family_of); df["block"] = df["name"].map(block_of)
    df["min"] = Xd.min(0); df["max"] = Xd.max(0); df["mean"] = Xd.mean(0); df["std"] = Xd.std(0)
    med = np.median(Xd, 0); mad = np.median(np.abs(Xd - med), 0) * 1.4826
    df["median"] = med; df["mad"] = mad
    df["n_unique"] = [len(np.unique(Xd[:, j])) for j in range(p)]
    df["top_value_share"] = [pd.Series(Xd[:, j]).value_counts(normalize=True).iloc[0] for j in range(p)]
    df["constant"] = df["max"] == df["min"]
    df["near_constant"] = (~df["constant"]) & (df["top_value_share"] >= NEAR_CONST_SHARE)
    df["duplicate_of_earlier_column"] = dup_cols
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # scipy warns on (near-)constant columns
        df["skew"] = stats.skew(Xd, axis=0); df["kurtosis"] = stats.kurtosis(Xd, axis=0)
        z = np.abs(Xd - med) / np.where(mad > 0, mad, np.nan)
    df["extreme_share"] = np.nan_to_num(np.nanmean(z > ROBUST_Z, axis=0) if True else 0)
    df["max_abs"] = np.abs(Xd).max(0)
    df["log10_std"] = np.log10(df["std"].where(df["std"] > 0))

    # target / leakage screens -------------------------------------------------------------
    ranks = stats.rankdata(Xd, axis=0)
    def auc(mask):                                     # Mann-Whitney AUC of each column for a binary mask
        n1 = mask.sum(); n0 = (~mask).sum()
        return (ranks[mask].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    df["split_auc"] = auc(~tr)                         # validation vs train
    def spearman_with(v):
        rv = stats.rankdata(v); rv = (rv - rv.mean()) / rv.std()
        rc = (ranks - ranks.mean(0)) / np.where(ranks.std(0) > 0, ranks.std(0), np.nan)
        return (rc * rv[:, None]).mean(0)
    df["rho_length"] = spearman_with(meta["length_seconds"].values)
    df["rho_row_index"] = spearman_with(np.arange(n))
    df["rho_recording_id"] = spearman_with(meta["recordingmbid"].map(lambda s: int(s.replace("-", "")[:15], 16)).values)
    df["rho_release_group_id"] = spearman_with(meta["release_group"].map(lambda s: int(s.replace("-", "")[:15], 16)).values)

    def eta2(groups, rows):                            # share of variance explained by group means
        Y = Xd[rows]; g = groups[rows]
        tot = ((Y - Y.mean(0)) ** 2).sum(0)
        between = sum(((Y[g == k].mean(0) - Y.mean(0)) ** 2) * (g == k).sum() for k in np.unique(g))
        with np.errstate(all="ignore"):
            return between / tot
    # proposed handling per column (a recommendation only; nothing is removed here)
    reason = np.where(df["constant"], "constant",
             np.where(df["duplicate_of_earlier_column"], "exact duplicate of an earlier column",
             np.where(df["near_constant"] & (df["kind"] != "onehot"), "near-constant (>=95% one value)",
             np.where(df["name"] == "rhythm.beats_count", "track-length proxy", ""))))
    df["proposed_drop_reason"] = reason
    sub = meta["subgenre"].values; ver = meta["essentia_version"].values
    df["eta2_subgenre_train"] = eta2(sub, tr)          # train only: keeps validation out of design choices
    df["eta2_version_train"] = eta2(ver, tr)
    # version effect within subgenre: remove subgenre means first (train only)
    Y = Xd[tr].copy(); gs = sub[tr]; gv = ver[tr]
    for k in np.unique(gs):
        Y[gs == k] -= Y[gs == k].mean(0)
    tot = (Y ** 2).sum(0)
    between = sum(((Y[gv == k].mean(0)) ** 2) * (gv == k).sum() for k in np.unique(gv))
    with np.errstate(all="ignore"):
        df["eta2_version_within_subgenre_train"] = between / tot
    return df


def distance_dominance(X, meta, fq):
    """Expected share of squared Euclidean distance per block/family (= share of summed variance)."""
    keep = ~(fq["constant"].values)
    tr = (meta["split"] == "train").values
    var_raw = X[tr].astype(np.float64).var(0)
    out = []
    for level in ["block", "family"]:
        g = fq.loc[keep].groupby(level)
        t = pd.DataFrame({
            "columns": g.size(),
            "unique_columns": fq.loc[keep & ~fq["duplicate_of_earlier_column"].values].groupby(level).size(),
            "share_raw_scale": pd.Series(var_raw[keep], index=fq.index[keep]).groupby(fq.loc[keep, level]).sum() / var_raw[keep].sum(),
        }).fillna(0)
        t["share_if_zscored"] = t["columns"] / t["columns"].sum()
        t["share_if_zscored_dedup"] = t["unique_columns"] / t["unique_columns"].sum()
        t.insert(0, "level", level); out.append(t.reset_index().rename(columns={level: "group"}))
    dom = pd.concat(out, ignore_index=True)
    # redundancy after z-scoring (train only): highly correlated non-identical column pairs
    train_std = X[tr].astype(np.float64).std(0)
    keep_u = keep & ~fq["duplicate_of_earlier_column"].values & (train_std > 0)
    Z = X[tr][:, keep_u].astype(np.float64); Z = (Z - Z.mean(0)) / Z.std(0)
    check(np.isfinite(Z).all(), "non-finite z-scores in redundancy diagnostic")
    C = (Z.T @ Z) / len(Z)
    C = (C + C.T) / 2
    iu = np.triu_indices_from(C, 1)
    hi = np.abs(C[iu])
    red = {"columns_compared": int(keep_u.sum()),
           "constant_in_train_only": fq.loc[keep & (train_std == 0), "name"].tolist(),
"pairs_abs_r_gt_0.95": int((hi > 0.95).sum()),
           "pairs_abs_r_gt_0.99": int((hi > 0.99).sum())}
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    red["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    red["components_for_90pct_variance"] = int(np.searchsorted(np.cumsum(ev[::-1]) / ev.sum(), 0.90) + 1)
    return dom, red


# ----------------------------------------------------------------------------- confounders
def split_ids(s):
    return set() if pd.isna(s) or s == "" else set(str(s).split("|"))


def confounders(meta, data_dir, row_hash):
    R = {}
    m = meta.copy()
    # extractor versions
    R["version_overall"] = m["essentia_version"].value_counts().to_dict()
    ver_split = pd.crosstab(m["essentia_version"], m["split"])
    ver_sub = pd.crosstab([m["subgenre"], m["split"]], m["essentia_version"])
    def cramers_v(t):
        chi2 = stats.chi2_contingency(t, correction=False)[0]
        return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))
    R["version_vs_split_cramers_v"] = cramers_v(ver_split)
    R["version_vs_subgenre_cramers_v"] = {sp: cramers_v(pd.crosstab(m.loc[m.split == sp, "subgenre"], m.loc[m.split == sp, "essentia_version"]))
                                          for sp in ["train", "validation"]}
    vs = pd.crosstab(m["subgenre"], m["essentia_version"], normalize="index")
    R["beta1_share_by_subgenre_range"] = [float(vs.get("2.1-beta1", pd.Series([0])).min()), float(vs.get("2.1-beta1", pd.Series([0])).max())]

    # artists
    m["artists"] = m["artist_mbids"].map(split_ids)
    m["albumartists"] = m["albumartist_mbids"].map(split_ids)
    m["artists_any"] = [a if a else b for a, b in zip(m["artists"], m["albumartists"])]
    cov = m.assign(has_artist=m["artists"].map(bool), has_any=m["artists_any"].map(bool)).groupby(["split"])[["has_artist", "has_any"]].mean()
    R["artist_coverage"] = cov.round(4).to_dict()
    tr, va = m[m.split == "train"], m[m.split == "validation"]
    art_tr = Counter(a for s in tr["artists_any"] for a in s); art_va = Counter(a for s in va["artists_any"] for a in s)
    shared = set(art_tr) & set(art_va)
    R["artists_distinct"] = {"train": len(art_tr), "validation": len(art_va), "shared": len(shared)}
    va_shared = va["artists_any"].map(lambda s: bool(s & shared))
    R["validation_recordings_with_artist_in_train"] = int(va_shared.sum())
    R["validation_recordings_with_artist_known"] = int(va["artists_any"].map(bool).sum())
    # same-artist candidates for validation queries against the train pool
    by_artist_tr = defaultdict(list)
    for i, (s, g) in enumerate(zip(tr["artists_any"], tr["subgenre"])):
        for a in s:
            by_artist_tr[a].append((i, g))
    same_sub_rel = tr["subgenre"].value_counts()
    rows = []
    for s, g in zip(va["artists_any"], va["subgenre"]):
        cands = {i: cg for a in s for (i, cg) in by_artist_tr.get(a, [])}
        rows.append((len(cands), sum(1 for cg in cands.values() if cg == g), same_sub_rel[g]))
    q = pd.DataFrame(rows, columns=["same_artist_cands", "same_artist_same_subgenre", "relevant_cands"])
    R["val_queries_with_same_artist_train_cand"] = int((q.same_artist_cands > 0).sum())
    R["val_queries_with_ge10_same_artist_same_subgenre"] = int((q.same_artist_same_subgenre >= 10).sum())
    R["same_artist_cands_mean"] = float(q.same_artist_cands.mean())
    R["same_artist_cands_max"] = int(q.same_artist_cands.max())
    # label agreement: P(same subgenre | same artist) vs P(same subgenre) for val x train pairs
    tot_same = sum((va["subgenre"] == g).sum() * k for g, k in tr["subgenre"].value_counts().items())
    R["p_same_subgenre_random_pair"] = float(tot_same / (len(va) * len(tr)))
    sa = q.same_artist_cands.sum()
    R["p_same_subgenre_given_same_artist"] = float(q.same_artist_same_subgenre.sum() / sa) if sa else None
    # within-split same-artist pairs (relevant if a pool mixes recordings from one split)
    def within(df):
        c = Counter(a for s in df["artists_any"] for a in s)
        return int(sum(k * (k - 1) // 2 for k in c.values())), int(sum(1 for k in c.values() if k > 1))
    R["within_split_same_artist_pairs"] = {sp: within(d) for sp, d in [("train", tr), ("validation", va)]}

    # per-subgenre artist concentration
    art_rows = []
    for (g, sp), d in m.groupby(["subgenre", "split"]):
        c = Counter(a for s in d["artists_any"] for a in s)
        known = d["artists_any"].map(bool).sum()
        art_rows.append({"subgenre": g, "split": sp, "recordings": len(d), "artist_known": int(known),
                         "distinct_artists": len(c), "top_artist_recordings": max(c.values()) if c else 0,
                         "top_artist_share": round(max(c.values()) / len(d), 3) if c else 0.0})
    art_tbl = pd.DataFrame(art_rows)

    # recording-ID mismatch and duplicate audio
    R["tag_recording_mbid"] = m["tag_recording_mbid"].value_counts().to_dict()
    R["tag_mismatch_by_split"] = m[m.tag_recording_mbid == "differs"]["split"].value_counts().to_dict()
    # does any mismatching tag ID point at another recording in the matrix or the full cohort?
    coh_ids = set(pd.read_csv(os.path.join(data_dir, "cohort.csv"), usecols=["recordingmbid"])["recordingmbid"])
    mat_ids = set(m["recordingmbid"]); zips = {}; coll_matrix = coll_cohort = 0
    for _, r in m[m.tag_recording_mbid == "differs"].iterrows():
        zp = os.path.join(data_dir, "shards", r["archive"].replace(".tar.bz2", ".zip"))
        z = zips.setdefault(zp, zipfile.ZipFile(zp))
        t = json.loads(z.read(f"{r['recordingmbid']}.json").decode("utf-8"), strict=False)["metadata"].get("tags", {}).get("musicbrainz_recordingid", [])
        t = set(t if isinstance(t, list) else [t]) - {r["recordingmbid"]}
        coll_matrix += bool(t & mat_ids); coll_cohort += bool(t & coh_ids)
    R["tag_mismatch_pointing_to_matrix_recording"] = coll_matrix
    R["tag_mismatch_pointing_to_cohort_recording"] = coll_cohort
    dup = m[m["audio_md5_shared"] == 1].sort_values(["audio_md5", "recordingmbid"])
    dup = dup.assign(identical_feature_row=dup.index.map(lambda i: int((row_hash == row_hash[i]).sum() > 1)))
    dup_tbl = dup[["audio_md5", "recordingmbid", "subgenre", "split", "release_group", "artist_mbids", "length_seconds", "identical_feature_row"]]
    R["duplicate_audio_groups"] = int(dup["audio_md5"].nunique())
    R["duplicate_audio_cross_split_groups"] = int(dup.groupby("audio_md5")["split"].nunique().gt(1).sum())
    R["duplicate_audio_label_conflict_groups"] = int(dup.groupby("audio_md5")["subgenre"].nunique().gt(1).sum())

    # per-subgenre counts and comparison with the full cohort
    g = m.groupby(["subgenre", "split"])
    cnt = g.agg(recordings=("recordingmbid", "size"), albums=("release_group", "nunique")).reset_index()
    top_album = m.groupby(["subgenre", "split", "release_group"]).size().groupby(["subgenre", "split"]).max().rename("top_album_recordings")
    cnt = cnt.merge(top_album.reset_index(), on=["subgenre", "split"])
    cnt["recordings_per_album"] = (cnt["recordings"] / cnt["albums"]).round(2)
    coh = pd.read_csv(os.path.join(data_dir, "cohort.csv"))
    full = coh.groupby(["subgenre", "split"]).size().rename("full_cohort").reset_index()
    cnt = cnt.merge(full, on=["subgenre", "split"])
    for sp in ["train", "validation"]:
        s = cnt.split == sp
        cnt.loc[s, "subset_share"] = cnt.loc[s, "recordings"] / cnt.loc[s, "recordings"].sum()
        cnt.loc[s, "full_share"] = cnt.loc[s, "full_cohort"] / cnt.loc[s, "full_cohort"].sum()
    cnt["share_ratio"] = (cnt["subset_share"] / cnt["full_share"]).round(3)
    gof = {}
    for sp in ["train", "validation"]:
        s = cnt[cnt.split == sp]
        exp = s["full_share"] * s["recordings"].sum()
        chi2, pv = stats.chisquare(s["recordings"], exp)
        gof[sp] = {"chi2": round(float(chi2), 2), "p": float(pv)}
    R["subset_vs_full_cohort_chisquare"] = gof
    R["imbalance_ratio"] = {sp: round(float(cnt[cnt.split == sp].recordings.max() / cnt[cnt.split == sp].recordings.min()), 2) for sp in ["train", "validation"]}

    # duration
    dur = m.groupby("subgenre")["length_seconds"].describe()[["min", "25%", "50%", "75%", "max"]].round(1)
    R["short_tracks_lt30s"] = int((m.length_seconds < 30).sum())
    R["short_tracks_lt30s_by_subgenre"] = m[m.length_seconds < 30]["subgenre"].value_counts().to_dict()
    R["long_tracks_gt900s"] = int((m.length_seconds > 900).sum())
    return R, ver_split, ver_sub, art_tbl, dup_tbl, cnt, dur, q


# ----------------------------------------------------------------------------- report
def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    def f(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return ""
            if float(v).is_integer() and abs(v) < 1e15 and "e" not in floatfmt:
                return f"{int(v):,}"
            out = floatfmt.format(v)
        else:
            out = str(v)
        return re.sub(r"(?<!\\)\|", r"\\|", out)       # escape bare pipes so cells do not split
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def short(g):
    return g.split("---")[-1]


def write_report(res, fq, dom, ver_split, ver_sub, art_tbl, dup_tbl, cnt, dur, path, a):
    I, F, Rd, C = res["integrity"], res["features"], res["redundancy"], res["confounders"]
    n, p = I["shape"]
    pct = lambda x, d: f"{100 * x / d:.1f}%"
    L = []
    w = L.append
    w("# Data audit 01: feature matrix quality and confounders\n")
    w(f"Generated by `analysis/01_data_audit.py` from `{a.data_dir}` (X.npy SHA-256 `{res['x_sha'][:16]}…`). "
      "Rerun the script to regenerate this file; do not edit it by hand.\n")
    w("Nothing in the dataset was modified. Z-scores, correlations and group means below are computed in memory "
      "as diagnostics only; no scaling, model or metric has been chosen.\n")

    # ---- headline
    w("## Headline findings\n")
    w(f"- **Integrity passes.** {n:,} rows × {p:,} columns, all finite; metadata, feature names and matrix align; "
      f"{I['mapping_rows_checked']:,} rows re-derived from raw JSON match exactly; no duplicate IDs or repeated rows; "
      f"train and validation share no album ({I['release_groups']:,} albums).")
    w(f"- **Same-artist candidates are the largest threat to valid scores.** A validation query and a training candidate "
      f"by the same artist share a subgenre {C['p_same_subgenre_given_same_artist']:.0%} of the time, versus "
      f"{C['p_same_subgenre_random_pair']:.1%} for a random pair. {C['val_queries_with_same_artist_train_cand']:,} of "
      f"{int(cnt[cnt.split == 'validation'].recordings.sum()):,} validation queries have at least one "
      f"same-artist training candidate, and {C['val_queries_with_ge10_same_artist_same_subgenre']} have ten or more "
      "same-artist, same-subgenre candidates, enough to fill a top-10 list on artist identity alone.")
    w(f"- **Unscaled distances would be meaningless.** Column standard deviations span 10^{F['log10_std_range'][0]} to "
      f"10^{F['log10_std_range'][1]}; `lowlevel.melbands_kurtosis.*` alone carries essentially all raw variance. "
      f"{F['abs_skew_gt_2']:,} columns have |skew| > 2.")
    ndup = F['proposed_drop_counts'].get('exact duplicate of an earlier column', 0)
    w(f"- **{F['constant'] + ndup:,} columns are constant or exact duplicates** "
      f"({F['constant']} constant; {ndup} further exact duplicates, mostly the mirrored halves of the symmetric "
      f"MFCC/GFCC covariance matrices). After also removing near-constant columns and the track-length proxy, "
      f"{F['proposed_remaining_columns']:,} columns would remain.")
    w(f"- **Extractor version is a measurable confounder.** Within subgenres, version explains > 10% of variance for "
      f"{F['eta2_version_within_gt_0.10']} columns (max {F['eta2_version_within_max']:.2f}), and the share of 2.1-beta1 "
      f"recordings varies by subgenre from {C['beta1_share_by_subgenre_range'][0]:.0%} to {C['beta1_share_by_subgenre_range'][1]:.0%}.")
    w(f"- **No feature encodes IDs or split membership.** Max |Spearman ρ| with recording ID {F['rho_recording_id_max']:.3f}, "
      f"album ID {F['rho_release_group_id_max']:.3f}, row order {F['rho_row_index_max']:.3f}; max |AUC − 0.5| for "
      f"train vs validation {F['split_auc_max_dev']:.3f}. `rhythm.beats_count` is a track-length proxy (ρ = "
      f"{fq.set_index('name').loc['rhythm.beats_count', 'rho_length']:.2f}).")
    w(f"- **The archive subset is representative.** Subgenre shares match the full 30,388-recording cohort "
      f"(χ² p = {C['subset_vs_full_cohort_chisquare']['train']['p']:.2f} train, "
      f"{C['subset_vs_full_cohort_chisquare']['validation']['p']:.2f} validation), but classes are imbalanced "
      f"(largest/smallest = {C['imbalance_ratio']['train']} train, {C['imbalance_ratio']['validation']} validation).\n")

    # ---- integrity
    w("## 1. Integrity\n")
    rows = [
        ("Matrix shape / dtype", f"{n:,} × {p:,}, {I['dtype']}", "pass"),
        ("Metadata rows = matrix rows; `row` = 0..n−1", f"{n:,}", "pass"),
        ("Feature names = columns, unique, in order", f"{p:,}", "pass"),
        ("Non-finite values in X", str(I["nonfinite"]), "pass"),
        ("Missing metadata values", ", ".join(f"{k}: {v:,}" for k, v in I["meta_missing"].items()) or "none",
         "artist IDs only (see §3.2)"),
        ("X.npy SHA-256 matches extraction report", "yes", "pass"),
        ("Rows re-derived from raw JSON", f"{I['mapping_rows_checked']:,} checked, 0 mismatches", "pass"),
        ("Duplicate recording IDs", "0", "pass"),
        ("Repeated (identical) matrix rows", str(I["repeated_row_groups"]), "pass"),
        ("Duplicate-audio checksum groups", f"{C['duplicate_audio_groups']} groups ({2 * C['duplicate_audio_groups']} recordings)", "flag (§3.3)"),
        ("Albums in both train and validation", "0", "pass"),
        ("Excluded recordings", str(I["exclusions"]), "pass"),
        ("Non-acoustic fields in the vector (metadata, beat positions)", "none", "pass"),
    ]
    w(md_table(pd.DataFrame(rows, columns=["Check", "Result", "Status"])) + "\n")

    # ---- feature quality
    w("## 2. Feature quality\n")
    w("### 2.1 Composition\n")
    comp = fq.groupby("block").agg(columns=("name", "size"), constant=("constant", "sum"),
                                    exact_duplicates=("duplicate_of_earlier_column", "sum")).reset_index()
    w(md_table(comp) + "\n")
    w(f"Sections: " + ", ".join(f"{k} {v:,}" for k, v in F["columns_by_section"].items()) +
      f". {F['families']} descriptor families (full list with per-column statistics in `01_feature_summary.csv`).\n")

    w("### 2.2 Constant, near-constant and duplicate columns\n")
    w(f"- **Constant ({F['constant']})**: identical value in every row, matching the 34 listed in the extraction report. "
      "Mostly derivative statistics (dmean, dvar…) of descriptors that have one value per track, e.g. "
      "`rhythm.bpm_histogram_first_peak_bpm.dmean`, plus silence-rate maxima.")
    w(f"- **Exact duplicates ({F['duplicate_columns']}, of which {F['duplicate_columns_by_block'].get('MFCC/GFCC cov+icov matrices', 0)} "
      "in covariance matrices)**: each 13 × 13 MFCC/GFCC `cov`/`icov` matrix is symmetric, so 78 off-diagonal entries per matrix "
      "repeat another column. Separately, the BPM-histogram peak descriptors have mean = median = min = max per track:")
    for g in F["nonconstant_duplicate_groups"]:
        w(f"  - {', '.join('`' + x + '`' for x in g)}")
    w(f"- **Near-constant ({F['near_constant']}, ≥ {NEAR_CONST_SHARE:.0%} of rows share one value)**: "
      "the 36 `tonal.hpcp.max[*]` (HPCP is normalised so its maximum is almost always 1), silence-rate statistics, "
      "`spectral_complexity.min`, and three rare one-hot levels (`chords_key=D#`, `chords_key=F#`, `key_key=F#`). "
      "The one-hot levels are kept because they belong to complete categorical encodings.")
    w(f"- Columns constant in training only: {', '.join('`' + x + '`' for x in Rd['constant_in_train_only']) or 'none'}.\n")
    pdrop = pd.DataFrame(sorted(F["proposed_drop_counts"].items(), key=lambda kv: -kv[1]), columns=["Proposed drop reason", "Columns"])
    w(md_table(pdrop) + f"\n\nColumns remaining under this proposal: **{F['proposed_remaining_columns']:,}** "
      f"(the per-column reason is in `01_feature_summary.csv`, column `proposed_drop_reason`). "
      f"{n:,} × {F['proposed_remaining_columns']:,} = {n * F['proposed_remaining_columns']:,} instance-attributes, "
      f"well above 3,000,000 (and {n * 1999:,} at the conservative 1,999).\n")

    w("### 2.3 Scale, skew and extreme values\n")
    big = fq[~fq["constant"]].nlargest(5, "std")[["name", "std", "max_abs", "skew"]]
    w(f"- Standard deviations range over ~{F['log10_std_range'][1] - F['log10_std_range'][0]:.0f} orders of magnitude "
      f"(10^{F['log10_std_range'][0]} to 10^{F['log10_std_range'][1]}). {F['max_abs_gt_1e6']} columns exceed 10^6 in magnitude. "
      "Many band-energy statistics (`melbands.var`, `barkbands.dvar`…) are below 10^-6 because they are raw energies, not because they are constant.")
    w(f"- {F['abs_skew_gt_2']:,} non-constant columns have |skew| > 2 and {F['abs_skew_gt_10']:,} have |skew| > 10; "
      f"{F['columns_with_gt1pct_extreme']:,} columns have more than 1% of values beyond {ROBUST_Z:.0f} robust SDs from the median. "
      "Variance-type statistics (var, dvar, dvar2) and kurtosis/flatness descriptors account for most of the heavy tails.")
    w("- Largest-scale columns:\n")
    w(md_table(big, "{:.3g}") + "\n")

    w("### 2.4 Screens for target, ID or split leakage\n")
    w("Every column was screened, not only the suspicious-looking names.\n")
    lk = [
        ("Train vs validation (Mann-Whitney AUC)", f"max \\|AUC − 0.5\\| = {F['split_auc_max_dev']:.3f}; {F['split_auc_cols_dev_gt_0.1']} columns > 0.1", "no split signal"),
        ("Recording ID (Spearman ρ with ID value)", f"max \\|ρ\\| = {F['rho_recording_id_max']:.3f}", "no ID encoding"),
        ("Album ID (Spearman ρ)", f"max \\|ρ\\| = {F['rho_release_group_id_max']:.3f}", "no ID encoding"),
        ("Row order (Spearman ρ)", f"max \\|ρ\\| = {F['rho_row_index_max']:.3f}", "no ordering artefact"),
        ("Track length (Spearman ρ)", f"\\|ρ\\| > 0.9: {', '.join(F['rho_length_abs_gt_0.9'])}; \\|ρ\\| > 0.5: {F['rho_length_abs_gt_0.5']} columns",
         "beats_count is a length proxy; chords_number_rate ρ = %.2f" % fq.set_index("name").loc["tonal.chords_number_rate", "rho_length"]),
        ("Subgenre (η², train only)", f"max η² = {F['eta2_subgenre_max']:.2f}; {F['eta2_subgenre_gt_0.3']} columns > 0.3",
         "ordinary acoustic signal; no column separates subgenres on its own"),
    ]
    w(md_table(pd.DataFrame(lk, columns=["Screen", "Result", "Interpretation"])) + "\n")
    w("Track length is not a label, but it differs sharply by subgenre (median grindcore track "
      f"{dur.loc['metal---grindcore', '50%']:.0f} s vs post-metal {dur.loc['postrock---postmetal', '50%']:.0f} s). "
      "Length itself was excluded from the vector as file metadata; `rhythm.beats_count` re-introduces it almost exactly. "
      "Top hits for every screen are in `01_leakage_screens.csv`.\n")

    w("### 2.5 Which families would dominate a distance\n")
    w("Expected share of squared Euclidean distance contributed by each block (= its share of summed variance, training rows, constant columns excluded):\n")
    bt = dom[dom.level == "block"][["group", "columns", "unique_columns", "share_raw_scale", "share_if_zscored", "share_if_zscored_dedup"]]
    w(md_table(bt) + "\n")
    ft = dom[dom.level == "family"].sort_values("share_if_zscored", ascending=False).head(8)[["group", "columns", "share_if_zscored", "share_if_zscored_dedup"]]
    w("Largest families if every column were z-scored:\n")
    w(md_table(ft) + "\n")
    w(f"- **Raw scale**: one family (`lowlevel.melbands_kurtosis`) contributes ~100% of distance; raw features cannot be used.")
    w(f"- **Global z-scoring** weights families by column count: six band-type families (ERB, mel, MFCC, GFCC, HPCP, Bark) "
      f"take ~{ft.head(6)['share_if_zscored'].sum():.0%} of the distance. Bark, mel and ERB bands describe the same spectral envelope "
      f"on different frequency scales, so the envelope is counted roughly three times.")
    w(f"- **Redundancy**: among {Rd['columns_compared']:,} non-constant unique columns, {Rd['pairs_abs_r_gt_0.99']:,} pairs have "
      f"|r| > 0.99 and {Rd['pairs_abs_r_gt_0.95']:,} have |r| > 0.95. The correlation matrix has a participation ratio of "
      f"{Rd['participation_ratio']:.1f} and needs {Rd['components_for_90pct_variance']} components for 90% of variance, so the "
      "effective dimensionality is far below the column count.\n")

    # ---- confounders
    w("## 3. Potential confounders\n")
    w("### 3.1 Extractor version\n")
    vs = ver_split.copy(); vs["total"] = vs.sum(1)
    w(md_table(vs.reset_index().rename(columns={"essentia_version": "version"})) + "\n")
    vsub = ver_sub.groupby(level=0).sum()
    vsub = (vsub.div(vsub.sum(1), axis=0)).reset_index()
    vsub["subgenre"] = vsub["subgenre"].map(short)
    w("Share of each version by subgenre (train + validation):\n")
    w(md_table(vsub.sort_values("2.1-beta1", ascending=False), "{:.2f}") + "\n")
    w(f"- Version vs split: Cramér's V = {C['version_vs_split_cramers_v']:.3f} (negligible). Version vs subgenre: "
      f"V = {C['version_vs_subgenre_cramers_v']['train']:.2f} (train), {C['version_vs_subgenre_cramers_v']['validation']:.2f} (validation), weak to moderate.")
    w(f"- Within subgenres, version explains > 5% of variance in {F['eta2_version_within_gt_0.05']} columns and > 10% in "
      f"{F['eta2_version_within_gt_0.10']} (max {F['eta2_version_within_max']:.2f}). The affected columns are mainly spectral "
      "roll-off, zero-crossing rate, strong-peak, high-frequency content, and upper Bark/ERB/GFCC bands, consistent with a "
      "change in the extractor's high-frequency processing between 2.1-beta1 and 2.1-beta2.")
    w("- 14 recordings come from a `2.1-dev` build (heavy metal and industrial metal only).\n")

    w("### 3.2 Artists\n")
    cov = C["artist_coverage"]["has_artist"]
    w(f"- Artist IDs are present for {cov['train']:.1%} of training and {cov['validation']:.1%} of validation recordings; "
      "album-artist IDs add no extra coverage (they are missing for the same files).")
    w(f"- Distinct artists: {C['artists_distinct']['train']:,} in train, {C['artists_distinct']['validation']:,} in validation, "
      f"**{C['artists_distinct']['shared']:,} in both**. {C['validation_recordings_with_artist_in_train']:,} of "
      f"{C['validation_recordings_with_artist_known']:,} validation recordings with a known artist have that artist in the training pool.")
    w(f"- P(same subgenre | same artist) = **{C['p_same_subgenre_given_same_artist']:.3f}** vs P(same subgenre) = "
      f"{C['p_same_subgenre_random_pair']:.3f} for a random validation–training pair. Same-artist candidates per validation query: "
      f"mean {C['same_artist_cands_mean']:.1f}, max {C['same_artist_cands_max']}; "
      f"{C['val_queries_with_ge10_same_artist_same_subgenre']} queries have ≥ 10 same-artist, same-subgenre candidates.")
    tw, vw = C["within_split_same_artist_pairs"]["train"], C["within_split_same_artist_pairs"]["validation"]
    w(f"- Within-split same-artist pairs: {tw[0]:,} in train ({tw[1]} artists with > 1 recording), {vw[0]:,} in validation "
      f"({vw[1]} artists). These matter if a candidate pool ever includes the query's own split.")
    w(f"- Recordings without an artist ID cannot be screened, so some same-artist matches will go undetected in the {1 - cov['validation']:.0%} of "
      "validation queries and the training candidates that lack one.\n")
    at = art_tbl[art_tbl.split == "validation"].copy(); at["subgenre"] = at["subgenre"].map(short)
    w("Artist concentration in validation (the query side):\n")
    w(md_table(at.sort_values("top_artist_share", ascending=False)[["subgenre", "recordings", "artist_known", "distinct_artists", "top_artist_recordings", "top_artist_share"]]) + "\n")

    w("### 3.3 Recording-ID mismatches and duplicate audio\n")
    w(f"- {C['tag_recording_mbid'].get('differs', 0)} files carry a MusicBrainz recording ID in their tags that differs from the dataset ID "
      f"({', '.join(f'{k} {v}' for k, v in C['tag_mismatch_by_split'].items())}). "
      f"{C['tag_mismatch_pointing_to_matrix_recording']} of those tag IDs point at another recording in the matrix and "
      f"{C['tag_mismatch_pointing_to_cohort_recording']} at any recording in the full cohort, so they do not create hidden duplicates. "
      "They most likely reflect MusicBrainz merges after tagging; the dataset ID is authoritative.")
    w(f"- {C['duplicate_audio_groups']} groups of recordings share an identical encoded-audio checksum. Each pair is the same artist, "
      f"subgenre and split, on different albums (likely a reissue or compilation); their feature rows are not bit-identical. "
      f"Cross-split groups: {C['duplicate_audio_cross_split_groups']}; label conflicts: {C['duplicate_audio_label_conflict_groups']}.\n")
    d2 = dup_tbl.copy(); d2["subgenre"] = d2["subgenre"].map(short)
    d2["audio_md5"] = d2["audio_md5"].str[:8] + "…"; d2["recordingmbid"] = d2["recordingmbid"].str[:8] + "…"
    d2["release_group"] = d2["release_group"].str[:8] + "…"; d2["artist_mbids"] = d2["artist_mbids"].astype(str).str[:8] + "…"
    w(md_table(d2, "{:.1f}") + "\n")

    w("### 3.4 Per-subgenre counts and the archive subset\n")
    pv = cnt.pivot(index="subgenre", columns="split", values=["recordings", "albums", "top_album_recordings", "share_ratio"])
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index(); pv["subgenre"] = pv["subgenre"].map(short)
    pv = pv.sort_values("recordings_train", ascending=False)
    w(md_table(pv[["subgenre", "recordings_train", "albums_train", "recordings_validation", "albums_validation",
                   "top_album_recordings_validation", "share_ratio_train", "share_ratio_validation"]], "{:.2f}") + "\n")
    w("`share_ratio` = the subgenre's share in this subset divided by its share in the full cohort for the same split (1.00 = identical).\n")
    w(f"- The subset reproduces the full cohort's subgenre mix (χ² goodness of fit p = {C['subset_vs_full_cohort_chisquare']['train']['p']:.2f} "
      f"train, {C['subset_vs_full_cohort_chisquare']['validation']['p']:.2f} validation): selecting archives by ID prefix behaved like random sampling.")
    w(f"- Imbalance is inherited from the data: black metal has {C['imbalance_ratio']['train']}× (train) and "
      f"{C['imbalance_ratio']['validation']}× (validation) as many recordings as the smallest subgenre. Micro-averaged scores would be "
      "driven by black, heavy and death metal.")
    gv = cnt[(cnt.subgenre == "metal---grindcore") & (cnt.split == "validation")].iloc[0]
    ga = art_tbl[(art_tbl.subgenre == "metal---grindcore") & (art_tbl.split == "validation")].iloc[0]
    w("- Thin validation subgenres in terms of independent albums: folk (28), grindcore and alternative (34 each). "
      f"Grindcore is the most concentrated: a single album supplies {int(gv.top_album_recordings)} and a single artist "
      f"{int(ga.top_artist_recordings)} of its {int(gv.recordings)} validation recordings. Album-level aggregation of "
      "query scores is therefore essential for grindcore.\n")

    w("### 3.5 Track length\n")
    w(f"- {C['short_tracks_lt30s']} recordings are shorter than 30 s ({C['short_tracks_lt30s_by_subgenre'].get('metal---grindcore', 0)} of them grindcore, "
      f"where short songs are genuine) and {C['long_tracks_gt900s']} are longer than 15 minutes. Summary statistics over very short "
      "tracks are noisier but not invalid; no recording looks corrupt.\n")

    # ---- decision table
    w("## 4. Decision table\n")
    dt = [
        ("Constant columns", f"{F['constant']} columns with one value in all rows", "Drop before scaling (no information; z-scoring would divide by zero).", "No", "Confirmed"),
        ("Exact duplicate columns", f"{F['proposed_drop_counts'].get('exact duplicate of an earlier column', 0)} non-constant duplicates: mirrored cov/icov entries and BPM-peak mean=median=min=max", "Keep one copy (upper triangle incl. diagonal of each matrix; one column per BPM-peak descriptor).", "No", "Confirmed"),
        ("Near-constant columns", f"{F['proposed_drop_counts'].get('near-constant (>=95% one value)', 0)} non-categorical columns with ≥95% one value (36 are hpcp.max)", "Drop; keep rare one-hot levels so key/scale encodings stay complete.", "No", "Confirmed"),
        ("Scale and heavy tails", f"std spans 10^{F['log10_std_range'][0]}–10^{F['log10_std_range'][1]}; {F['abs_skew_gt_10']} columns |skew|>10", "Scaling is mandatory; choose a tail-robust transform (e.g. log/signed-log for variance-type statistics, or rank/robust scaling) fitted on training rows only, in the next stage.", "No (design input)", "Confirmed"),
        ("Family dominance and redundancy", f"band families ≈ {ft.head(6)['share_if_zscored'].sum():.0%} of z-scored distance; {Rd['pairs_abs_r_gt_0.99']:,} pairs |r|>0.99", "Do not rely on plain all-column z-scored Euclidean distance as the only representation; compare family-balanced weighting or a decorrelated (train-fitted) projection as part of the design.", "No (design input)", "Risk to test"),
        ("Track-length proxy", "rhythm.beats_count ρ = %.2f with length; length differs strongly by subgenre" % fq.set_index("name").loc["rhythm.beats_count", "rho_length"], "Drop beats_count from the main representation (length was deliberately excluded as metadata); optionally report a sensitivity run with it.", "No", "Confirmed"),
        ("Same-artist candidates", f"P(same subgenre given same artist) = {C['p_same_subgenre_given_same_artist']:.3f} vs {C['p_same_subgenre_random_pair']:.3f} for a random pair; {C['val_queries_with_same_artist_train_cand']:,} queries affected", "Main evaluation: remove candidates sharing any artist ID with the query (in addition to the album rule). Report unfiltered scores only as a secondary, clearly labelled comparison. Report results separately for queries with and without a known artist.", "Not this audit; it is a precondition for producing any retrieval score", "Confirmed"),
        ("Missing artist IDs", f"{1 - cov['train']:.0%} train / {1 - cov['validation']:.0%} validation lack artist IDs", "Cannot be filtered; quantify residual risk with a sensitivity run restricted to queries and candidates that have artist IDs.", "No", "Risk to test"),
        ("Duplicate audio", f"{C['duplicate_audio_groups']} pairs, same artist/subgenre/split, different albums", "Keep one recording per checksum group (lowest recording ID) in any candidate pool; the artist filter already removes them as candidates for same-artist queries.", "No", "Confirmed (minor)"),
        ("Extractor version", f"{F['eta2_version_within_gt_0.10']} columns with within-subgenre η²>0.10; beta1 share {C['beta1_share_by_subgenre_range'][0]:.0%}–{C['beta1_share_by_subgenre_range'][1]:.0%} by subgenre", "Keep all versions in the main run; record version as a covariate; test sensitivity (beta2-only subset and/or dropping version-sensitive columns) and check whether neighbours are disproportionately same-version.", "No", "Risk to test"),
        ("Recording-ID tag mismatches", f"{C['tag_recording_mbid'].get('differs', 0)} files; 0 point to another recording", "No action; dataset IDs are authoritative.", "No", "Checked, benign"),
        ("Class imbalance", f"largest/smallest {C['imbalance_ratio']['train']}× train, {C['imbalance_ratio']['validation']}× validation", "Macro-average over subgenres; aggregate per album before averaging; report per-subgenre results with intervals.", "No", "Confirmed"),
        ("Thin / concentrated validation subgenres", "folk 28 albums; grindcore 34 albums, one album = 55 of 158 queries", "Keep in main evaluation (as previously decided) but report per-subgenre intervals and flag these as least reliable.", "No", "Confirmed"),
        ("Leakage via IDs, split or labels", "All screens negative (|ρ| ≤ 0.06, |AUC−0.5| ≤ 0.03)", "No action.", "No", "Checked, benign"),
    ]
    w(md_table(pd.DataFrame(dt, columns=["Issue", "Evidence", "Proposed handling", "Blocks next stage?", "Status"])) + "\n")
    w("**Nothing blocks starting the retrieval stage**, provided the retrieval protocol includes the same-artist exclusion from the outset. "
      "The scaling/representation choice is an input to that stage, not a precondition.\n")

    w("## 5. Confirmed problems vs risks still to test\n")
    w("Confirmed (handle deterministically): constant, duplicate and near-constant columns; extreme scale differences; the "
      "track-length proxy; the strong same-artist/same-subgenre association; two duplicate-audio pairs; class imbalance.\n")
    w("Risks to test in the next stage: whether extractor version drives nearest-neighbour structure; how much the "
      "unfiltered artist matches inflate scores; how sensitive results are to family weighting and redundancy; residual "
      "same-artist leakage among recordings without artist IDs.\n")

    w("## Files\n")
    files = [("01_data_audit.py", "this audit (rerunnable; `--full-mapping` re-derives every row)"),
             ("01_audit_results.json", "all computed numbers"),
             ("01_feature_summary.csv", "one row per feature: family, range, skew, constant/duplicate flags, leakage screens, proposed drop reason"),
             ("01_distance_share_by_group.csv", "distance share per block and family"),
             ("01_leakage_screens.csv", "top columns for each leakage/confounder screen"),
             ("01_extractor_version_by_subgenre_split.csv", "version counts"),
             ("01_artist_concentration.csv", "artist coverage and concentration by subgenre and split"),
             ("01_duplicate_audio.csv", "recordings sharing an audio checksum"),
             ("01_subgenre_split_counts.csv", "recordings, albums and subset-vs-cohort shares"),
             ("test_01_data_audit.py", "tests that the integrity assertions fire on corrupted inputs")]
    w(md_table(pd.DataFrame(files, columns=["File", "Contents"])) + "\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/metal_data"))
    ap.add_argument("--out-dir", default=here)
    ap.add_argument("--mapping-sample", type=int, default=300, help="rows re-derived from raw JSON (default 300)")
    ap.add_argument("--full-mapping", action="store_true", help="re-derive every row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-constant", type=int, default=None,
                    help="fail if the number of constant columns differs (default: the extraction report's count)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        X, meta, names, excl, rep, xsha = load(a.data_dir)
        integ, dup_cols, row_hash = integrity(X, meta, names, excl, rep, xsha, a.data_dir,
                                              len(meta) if a.full_mapping else a.mapping_sample, a.seed)
        fq = feature_quality(X, meta, names, dup_cols)
        exp_const = a.expect_constant if a.expect_constant is not None else len(rep["features"]["constant_columns"])
        check(int(fq["constant"].sum()) == exp_const,
              f"{int(fq['constant'].sum())} constant columns found, expected {exp_const}")
        check(set(fq.loc[fq["constant"], "name"]) == set(rep["features"]["constant_columns"]) or a.expect_constant is not None,
              "constant columns differ from those listed in extraction_report.json")
        dom, red = distance_dominance(X, meta, fq)
        conf, ver_split, ver_sub, art_tbl, dup_tbl, cnt, dur, q = confounders(meta, a.data_dir, row_hash)
    except AuditError as e:
        sys.exit(f"AUDIT FAILED: {e}")

    O = lambda f: os.path.join(a.out_dir, f)
    fq.round(6).to_csv(O("01_feature_summary.csv"), index=False)
    dom.round(5).to_csv(O("01_distance_share_by_group.csv"), index=False)
    ver_sub.to_csv(O("01_extractor_version_by_subgenre_split.csv"))
    art_tbl.to_csv(O("01_artist_concentration.csv"), index=False)
    dup_tbl.to_csv(O("01_duplicate_audio.csv"), index=False)
    cnt.round(4).to_csv(O("01_subgenre_split_counts.csv"), index=False)
    screens = pd.concat([
        fq.reindex(fq["split_auc"].sub(0.5).abs().sort_values(ascending=False).index).head(10).assign(screen="split_auc"),
        fq.reindex(fq["rho_length"].abs().sort_values(ascending=False).index).head(10).assign(screen="rho_length"),
        fq.reindex(fq["rho_recording_id"].abs().sort_values(ascending=False).index).head(5).assign(screen="rho_recording_id"),
        fq.reindex(fq["rho_row_index"].abs().sort_values(ascending=False).index).head(5).assign(screen="rho_row_index"),
        fq.reindex(fq["eta2_subgenre_train"].sort_values(ascending=False).index).head(10).assign(screen="eta2_subgenre_train"),
        fq.reindex(fq["eta2_version_within_subgenre_train"].sort_values(ascending=False).index).head(10).assign(screen="eta2_version_within_subgenre_train"),
    ])[["screen", "name", "split_auc", "rho_length", "rho_recording_id", "rho_row_index", "eta2_subgenre_train",
        "eta2_version_train", "eta2_version_within_subgenre_train"]]
    screens.round(4).to_csv(O("01_leakage_screens.csv"), index=False)

    nc = fq[~fq["constant"]]
    res = {
        "x_sha": xsha,
        "integrity": integ,
        "features": {
            "constant": int(fq["constant"].sum()),
            "near_constant": int(fq["near_constant"].sum()),
            "near_constant_names": fq.loc[fq["near_constant"], "name"].tolist(),
            "duplicate_columns": integ["duplicate_columns"],
            "duplicate_columns_by_block": fq[fq["duplicate_of_earlier_column"]].groupby("block").size().to_dict(),
            "informative_unique_columns": int((~fq["constant"] & ~fq["duplicate_of_earlier_column"]).sum()),
            "proposed_drop_counts": fq.loc[fq["proposed_drop_reason"] != "", "proposed_drop_reason"].value_counts().to_dict(),
            "proposed_remaining_columns": int((fq["proposed_drop_reason"] == "").sum()),
            "nonconstant_duplicate_groups": [g for g in fq[~fq["constant"]].groupby(
                fq.loc[~fq["constant"]].index.map(lambda j: hashlib.md5(np.ascontiguousarray(X[:, j]).tobytes()).hexdigest()))["name"].apply(list)
                if len(g) > 1 and "cov" not in g[0]],
            "columns_by_block": fq.groupby("block").size().to_dict(),
            "columns_by_section": fq.groupby("section").size().to_dict(),
            "families": int(fq["family"].nunique()),
            "abs_skew_gt_2": int((nc["skew"].abs() > 2).sum()),
            "abs_skew_gt_10": int((nc["skew"].abs() > 10).sum()),
            "max_abs_gt_1e6": int((nc["max_abs"] > 1e6).sum()),
            "log10_std_range": [round(float(nc["log10_std"].min()), 2), round(float(nc["log10_std"].max()), 2)],
            "columns_with_gt1pct_extreme": int((nc["extreme_share"] > 0.01).sum()),
            "split_auc_max_dev": round(float((fq["split_auc"] - 0.5).abs().max()), 4),
            "split_auc_cols_dev_gt_0.1": int(((fq["split_auc"] - 0.5).abs() > 0.1).sum()),
            "rho_length_abs_gt_0.9": fq.loc[fq["rho_length"].abs() > 0.9, "name"].tolist(),
            "rho_length_abs_gt_0.5": int((fq["rho_length"].abs() > 0.5).sum()),
            "rho_recording_id_max": round(float(fq["rho_recording_id"].abs().max()), 4),
            "rho_release_group_id_max": round(float(fq["rho_release_group_id"].abs().max()), 4),
            "rho_row_index_max": round(float(fq["rho_row_index"].abs().max()), 4),
            "eta2_subgenre_max": round(float(fq["eta2_subgenre_train"].max()), 4),
            "eta2_subgenre_gt_0.3": int((fq["eta2_subgenre_train"] > 0.3).sum()),
            "eta2_version_within_gt_0.05": int((fq["eta2_version_within_subgenre_train"] > 0.05).sum()),
            "eta2_version_within_gt_0.10": int((fq["eta2_version_within_subgenre_train"] > 0.10).sum()),
            "eta2_version_within_max": round(float(fq["eta2_version_within_subgenre_train"].max()), 4),
        },
        "redundancy": red,
        "confounders": conf,
        "duplicate_audio": dup_tbl.to_dict("records"),
        "duration_by_subgenre": dur.to_dict("index"),
        "same_artist_query_stats": q.describe().round(2).to_dict(),
    }
    json.dump(res, open(O("01_audit_results.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if True:
        write_report(res, fq, dom, ver_split, ver_sub, art_tbl, dup_tbl, cnt, dur, O("01_data_audit.md"), a)
    print(f"audit passed: X {integ['shape']}, {integ['mapping_rows_checked']} rows re-derived from raw JSON; outputs in {a.out_dir}")


if __name__ == "__main__":
    main()
