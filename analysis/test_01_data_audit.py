#!/usr/bin/env python3
"""
Tests for analysis/01_data_audit.py.

Builds a small stratified copy of the real outputs (8 recordings per subgenre and split) in a temporary
directory, then checks that
  * the audit passes on the clean copy and writes its report, and
  * each integrity assertion fires, with a recognisable message, on a deliberately corrupted copy.
The real dataset in ~/metal_data is only read, never modified.

Usage:  python3 analysis/test_01_data_audit.py [--data-dir ~/metal_data]
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "01_data_audit.py")


def make_fixture(src, dst, per_group=8):
    F, G = os.path.join(src, "features"), os.path.join(dst, "features")
    os.makedirs(G)
    X = np.load(os.path.join(F, "X.npy"))
    meta = pd.read_csv(os.path.join(F, "metadata.csv"), dtype=str, keep_default_na=False)
    idx = meta.groupby(["subgenre", "split"], sort=False).head(per_group).index.sort_values()
    Xs, ms = X[idx], meta.loc[idx].reset_index(drop=True)
    ms["row"] = range(len(ms))
    save(G, Xs, ms)
    shutil.copy(os.path.join(F, "feature_names.csv"), G)
    shutil.copy(os.path.join(F, "exclusions.csv"), G)
    os.symlink(os.path.join(src, "shards"), os.path.join(dst, "shards"))          # read-only use
    os.symlink(os.path.join(src, "cohort.csv"), os.path.join(dst, "cohort.csv"))
    write_report(G, Xs, src)
    return Xs, ms


def save(G, X, meta):
    np.save(os.path.join(G, "X.npy"), X)
    meta.to_csv(os.path.join(G, "metadata.csv"), index=False)


def write_report(G, X, src):
    rep = json.load(open(os.path.join(src, "features", "extraction_report.json")))
    names = pd.read_csv(os.path.join(G, "feature_names.csv"))["name"]
    rep["X_shape"] = list(X.shape)
    rep["X_sha256"] = hashlib.sha256(open(os.path.join(G, "X.npy"), "rb").read()).hexdigest()
    rep["features"]["constant_columns"] = [names[j] for j in range(X.shape[1]) if np.ptp(X[:, j]) == 0]
    json.dump(rep, open(os.path.join(G, "extraction_report.json"), "w"))


def run(data_dir, out_dir, *extra):
    p = subprocess.run([sys.executable, AUDIT, "--data-dir", data_dir, "--out-dir", out_dir, *extra],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/metal_data"))
    a = ap.parse_args()
    real_sha = hashlib.sha256(open(os.path.join(a.data_dir, "features", "X.npy"), "rb").read()).hexdigest()
    results = []

    def case(name, corrupt, expect_msg, extra=()):
        with tempfile.TemporaryDirectory() as t:
            d, o = os.path.join(t, "data"), os.path.join(t, "out")
            X, m = make_fixture(a.data_dir, d)
            G = os.path.join(d, "features")
            if corrupt:
                corrupt(G, X, m)
            code, out = run(d, o, *extra)
            if expect_msg is None:
                ok = code == 0 and os.path.exists(os.path.join(o, "01_data_audit.md"))
            else:
                ok = code != 0 and "AUDIT FAILED" in out and expect_msg in out
            results.append((name, ok, out.strip().splitlines()[-1] if out.strip() else ""))

    def drop_meta_row(G, X, m):
        save(G, X, m.iloc[:-1])
    def nan_value(G, X, m):
        X = X.copy(); X[3, 10] = np.nan; save(G, X, m); write_report(G, X, a.data_dir)
    def edit_without_report(G, X, m):
        X = X.copy(); X[0, 0] += 1; save(G, X, m)
    def swap_ids(G, X, m):
        m = m.copy(); tr = m.index[m.split == "train"][:2]
        m.loc[tr, "recordingmbid"] = m.loc[tr[::-1], "recordingmbid"].values; save(G, X, m)
    def album_across_splits(G, X, m):
        m = m.copy(); v = m.index[m.split == "validation"][0]; t = m.index[m.split == "train"][0]
        m.loc[v, "release_group"] = m.loc[t, "release_group"]; save(G, X, m)
    def duplicate_id(G, X, m):
        m = m.copy(); m.loc[1, "recordingmbid"] = m.loc[0, "recordingmbid"]; save(G, X, m)
    def metadata_feature(G, X, m):
        n = pd.read_csv(os.path.join(G, "feature_names.csv")); n.loc[5, "name"] = "metadata.audio_properties.length"
        n.to_csv(os.path.join(G, "feature_names.csv"), index=False)
    def wrong_constant_count(G, X, m):
        pass

    case("clean fixture passes and writes the report", None, None)
    case("metadata row missing", drop_meta_row, "metadata has")
    case("NaN in the matrix", nan_value, "non-finite values in X")
    case("matrix edited after the extraction report", edit_without_report, "SHA-256 does not match")
    case("two recording IDs swapped between rows", swap_ids, "do not match their raw JSON", ("--full-mapping",))
    case("album present in both splits", album_across_splits, "appear in both splits")
    case("duplicate recording ID", duplicate_id, "duplicate recording IDs")
    case("metadata field in the feature vector", metadata_feature, "non-acoustic fields")
    case("unexpected number of constant columns", wrong_constant_count, "constant columns found, expected 999", ("--expect-constant", "999"))

    unchanged = hashlib.sha256(open(os.path.join(a.data_dir, "features", "X.npy"), "rb").read()).hexdigest() == real_sha
    results.append(("real dataset X.npy unchanged", unchanged, ""))
    width = max(len(r[0]) for r in results)
    for name, ok, last in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {last[:110]}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
