#!/usr/bin/env python3
"""
Stage 2: extract AcousticBrainz low-level features for the Last.fm single-subgenre metal cohort.

Steps (each can be run on its own; `run` does all three):
  cohort  Build the cohort from the Last.fm annotation TSVs, using the role table in ab_metal_audit.py:
          recordings with exactly one genuine metal subgenre, restricted to the 17 subgenres that pass
          the audit's strict rule (>= 500 recordings and >= 100 release groups, train + validation).
  fetch   For each official feature archive (Zenodo record 2553414), one at a time:
          download with parallel ranged requests (resumable, MD5-checked), stream it through bzip2 + tar
          without unpacking to disk, keep only cohort recordings' raw JSON in a small zip "shard",
          then delete the archive. Finished archives are recorded and skipped on rerun.
  build   Parse the retained JSON, check every recording against one fixed schema, and write
          X.npy (float32), metadata.csv, feature_names.csv, exclusions.csv and a report.

Feature vector
  Included: every fixed-length numeric field under lowlevel, rhythm and tonal (scalars, vectors,
            matrices flattened row-major), plus one-hot encodings of tonal.key_key / chords_key
            (12 pitch classes) and tonal.key_scale / chords_scale (major, minor).
  Excluded: the whole `metadata` section (identifiers, tags, file/codec properties, length, replay gain,
            extractor version) and rhythm.beats_position (variable length). Genre labels are never read
            from the JSON. Identifiers and labels are kept only in metadata.csv.

Usage
  python3 extract_features.py run --data-dir ~/metal_data
  python3 extract_features.py build --data-dir ~/metal_data            (re-build from shards only)
  python3 extract_features.py run --data-dir ~/metal_data --archives train-89,validation-01234567
          (download and process only these archives; build uses only the recordings they contain.
           Pass the same --archives to a later `build`.)
Testing hooks: --cohort-file, --local-archive (repeatable), --no-delete.
"""
import argparse, bz2, csv, hashlib, io, json, math, os, random, shutil, subprocess, sys, tarfile, threading
import time, urllib.error, urllib.request, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_metal_audit as A  # role table (R, GENUINE), label normalisation, file names

ZENODO_API = "https://zenodo.org/api/records/2553414"
ZENODO_FILE = "https://zenodo.org/records/2553414/files/{}?download=1"
SPLITS = ("train", "validation")
STRICT_MIN_REC, STRICT_MIN_RG = 500, 100
EXPECTED = {"subgenres": 17, "train": 24861, "validation": 5527}   # from the audit

SECTIONS = ("lowlevel", "rhythm", "tonal")
EXCLUDED_PATHS = {"rhythm.beats_position"}
PITCHES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
CATEGORICAL = {"tonal.key_key": PITCHES, "tonal.chords_key": PITCHES,
               "tonal.key_scale": ["major", "minor"], "tonal.chords_scale": ["major", "minor"]}
F32_MAX = 3.4e38
GiB = 2 ** 30


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- cohort
def build_cohort(data_dir, tsv_dir):
    per = {}
    dup = 0
    for sp in SPLITS:
        path = os.path.join(tsv_dir, A.FNAME.format(s="lastfm", p=sp))
        with bz2.open(path, "rt", encoding="utf-8", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            if next(r)[0] != "recordingmbid":
                sys.exit(f"unexpected header in {path}")
            for row in r:
                if len(row) < 2 or not row[0]:
                    continue
                labs = {A.norm(x) for x in row[2:] if x.strip()}
                g = [l for l in labs if A.R.get(("lastfm", l), ("?",))[0] in A.GENUINE]
                if len(g) != 1:
                    continue
                if row[0] in per:
                    dup += 1
                per[row[0]] = (row[1], sp, g[0])
    if dup:
        sys.exit(f"{dup} recording IDs appear more than once in the Last.fm TSVs")
    agg = defaultdict(lambda: [0, set()])
    for rec, (rg, sp, lab) in per.items():
        agg[lab][0] += 1; agg[lab][1].add(rg)
    strict = sorted(l for l, (n, rgs) in agg.items() if n >= STRICT_MIN_REC and len(rgs) >= STRICT_MIN_RG)
    rows = sorted([rec, rg, sp, lab] for rec, (rg, sp, lab) in per.items() if lab in strict)
    rows.sort(key=lambda r: (SPLITS.index(r[2]), r[0]))
    n = Counter(r[2] for r in rows)
    log(f"cohort: {len(strict)} subgenres, train {n['train']}, validation {n['validation']}")
    if (len(strict), n["train"], n["validation"]) != (EXPECTED["subgenres"], EXPECTED["train"], EXPECTED["validation"]):
        sys.exit(f"cohort does not match the audit ({EXPECTED}); refusing to continue")
    path = os.path.join(data_dir, "cohort.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["recordingmbid", "release_group", "split", "subgenre"]); w.writerows(rows)
    os.replace(path + ".tmp", path)
    return path


def read_cohort(path):
    with open(path, newline="") as f:
        return {r["recordingmbid"]: r for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------- archives
def archive_split_and_prefixes(key):
    stem = key[: -len(".tar.bz2")]
    split = "validation" if "-validation-" in stem else "train"
    return split, stem.rsplit("-", 1)[1]


def list_archives(data_dir, local, select=None):
    """All 10 official feature archives, or only those named in `select` (full Zenodo key or
    short form such as 'train-89' / 'validation-01234567')."""
    if local:
        out = []
        for p in local:
            key = os.path.basename(p)
            out.append({"key": key, "size": os.path.getsize(p), "md5": None, "local": os.path.abspath(p)})
        return out
    mpath = os.path.join(data_dir, "zenodo_manifest.json")
    if os.path.exists(mpath):
        files = json.load(open(mpath))
    else:
        with urllib.request.urlopen(ZENODO_API, timeout=60) as r:
            files = json.load(r)["files"]
        atomic_write(mpath, json.dumps(files, indent=1))
    out = []
    for f in files:
        if "-features-" in f["key"] and f["key"].endswith(".tar.bz2"):
            out.append({"key": f["key"], "size": f["size"], "md5": f["checksum"].split(":", 1)[1], "local": None})
    # validation first (smaller, checks the whole pipeline early), then train in prefix order
    out.sort(key=lambda a: (archive_split_and_prefixes(a["key"])[0] != "validation", archive_split_and_prefixes(a["key"])[1]))
    if len(out) != 10:
        sys.exit(f"expected 10 feature archives on Zenodo, found {len(out)}")
    if select:
        chosen = []
        for name in select:
            hits = [ar for ar in out if name == ar["key"]
                    or name == "-".join(archive_split_and_prefixes(ar["key"]))]
            if len(hits) != 1:
                sys.exit(f"--archives: {name!r} does not name exactly one archive; valid short names: "
                         + ", ".join("-".join(archive_split_and_prefixes(ar["key"])) for ar in out))
            if hits[0] not in chosen:
                chosen.append(hits[0])
        out = [ar for ar in out if ar in chosen]          # keep validation-first order
    return out


def restrict_cohort(cohort, archives):
    """Keep only cohort recordings whose ID prefix places them in one of `archives`."""
    keep = set()
    for ar in archives:
        keep |= cohort_for_archive(cohort, ar["key"])
    return {m: r for m, r in cohort.items() if m in keep}


def cohort_for_archive(cohort, key):
    split, prefixes = archive_split_and_prefixes(key)
    return {m for m, r in cohort.items() if r["split"] == split and m[0] in prefixes}


# ----------------------------------------------------------------------------- download
class Downloader:
    def __init__(self, conns=8, chunk=32 * 2 ** 20, margin=2.5 * GiB):
        self.conns, self.chunk, self.margin = conns, chunk, margin

    def remaining_bytes(self, dst, size):
        if os.path.exists(dst):
            return 0
        state = dst + ".chunks.json"
        done = set(json.load(open(state))) if os.path.exists(state) else set()
        n = math.ceil(size / self.chunk)
        return sum(min(self.chunk, size - i * self.chunk) for i in range(n) if i not in done)

    def has_space(self, dst, size):
        return shutil.disk_usage(os.path.dirname(dst)).free >= self.remaining_bytes(dst, size) + self.margin

    def _fetch(self, url, fd, i, size):
        a = i * self.chunk; b = min(size, a + self.chunk) - 1
        off = a      # bytes before `off` are already written; a retry continues from there
        for attempt in range(10):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{b}", "User-Agent": "metal-feature-extract/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206 or not (r.headers.get("Content-Range") or "").startswith(f"bytes {off}-{b}/"):
                        raise IOError(f"bad range response {r.status} {r.headers.get('Content-Range')}")
                    while True:
                        buf = r.read(2 ** 20)
                        if not buf:
                            break
                        if off + len(buf) > b + 1:
                            raise IOError("server sent more bytes than requested")
                        os.pwrite(fd, buf, off); off += len(buf)
                if off != b + 1:
                    raise IOError(f"short read, {off - a} of {b - a + 1} bytes so far; resuming")
                return i
            except Exception as e:  # network errors, 429/5xx, truncation
                wait = min(120, 5 * 2 ** attempt) + random.uniform(0, 3)
                if isinstance(e, urllib.error.HTTPError) and e.headers.get("Retry-After", "").isdigit():
                    wait = max(wait, int(e.headers["Retry-After"]))
                log(f"  chunk {i} attempt {attempt + 1} failed ({e}); retrying in {wait:.0f}s")
                time.sleep(wait)
        raise IOError(f"chunk {i} failed after 10 attempts")

    def get(self, key, size, md5, dst):
        if os.path.exists(dst):
            return dst
        if not self.has_space(dst, size):
            raise IOError(f"not enough free disk for {key}: need {self.remaining_bytes(dst, size) / GiB:.2f} GiB "
                          f"+ {self.margin / GiB:.1f} GiB margin, have {shutil.disk_usage(os.path.dirname(dst)).free / GiB:.2f}")
        part, state = dst + ".part", dst + ".chunks.json"
        done = set(json.load(open(state))) if os.path.exists(state) and os.path.exists(part) else set()
        fd = os.open(part, os.O_RDWR | os.O_CREAT)
        try:
            os.ftruncate(fd, size)
            n = math.ceil(size / self.chunk)
            todo = [i for i in range(n) if i not in done]
            log(f"download {key}: {size / 1e9:.2f} GB, {len(done)}/{n} chunks already done")
            t0, got, lock = time.time(), 0, threading.Lock()
            with ThreadPoolExecutor(self.conns) as ex:
                futs = [ex.submit(self._fetch, ZENODO_FILE.format(key), fd, i, size) for i in todo]
                for k, fu in enumerate(as_completed(futs), 1):
                    i = fu.result()
                    with lock:
                        os.fsync(fd)
                        done.add(i); got += min(self.chunk, size - i * self.chunk)
                        atomic_write(state, json.dumps(sorted(done)))
                    if k % 10 == 0 or k == len(todo):
                        el = time.time() - t0
                        log(f"  {key}: {len(done)}/{n} chunks, {got / el / 1e6:.2f} MB/s, "
                            f"eta {(size * (n - len(done)) / n) / max(got / el, 1) / 60:.0f} min")
        finally:
            os.close(fd)
        h = hashlib.md5()
        with open(part, "rb") as f:
            for blk in iter(lambda: f.read(8 * 2 ** 20), b""):
                h.update(blk)
        if h.hexdigest() != md5:
            os.remove(part); os.remove(state)
            raise IOError(f"MD5 mismatch for {key}; partial download discarded")
        os.replace(part, dst); os.remove(state)
        log(f"download {key}: MD5 verified")
        return dst


# ----------------------------------------------------------------------------- filter one archive
def filter_archive(path, key, want, split, shard_dir):
    """Stream the archive; keep only cohort members' raw JSON. Returns stats dict."""
    zpath = os.path.join(shard_dir, key.replace(".tar.bz2", ".zip"))
    st = {"archive": key, "split": split, "members": 0, "json_members": 0, "cohort_expected": len(want),
          "kept": 0, "duplicate_members": [], "wrong_location": [], "size_mismatch": [], "bytes_kept": 0}
    seen = set()
    t0 = time.time()
    proc = subprocess.Popen(["bzip2", "-dc", path], stdout=subprocess.PIPE, bufsize=2 ** 20)
    try:
        with zipfile.ZipFile(zpath + ".part", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z, \
                tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                st["members"] += 1
                parts = m.name.split("/")
                if not parts[-1].endswith(".json"):
                    continue
                st["json_members"] += 1
                mbid = parts[-1][:-5]
                if mbid not in want:
                    continue
                if len(parts) < 3 or parts[-3] != f"acousticbrainz-mediaeval-{split}" or parts[-2] != mbid[:2]:
                    st["wrong_location"].append(m.name)
                data = tf.extractfile(m).read()
                if len(data) != m.size:
                    st["size_mismatch"].append(mbid)
                    continue
                if mbid in seen:
                    st["duplicate_members"].append(mbid)
                    z.writestr(f"{mbid}.dup{st['duplicate_members'].count(mbid)}.json", data)
                    continue
                seen.add(mbid)
                z.writestr(f"{mbid}.json", data)
                st["kept"] += 1; st["bytes_kept"] += len(data)
                if st["kept"] % 1000 == 0:
                    log(f"  {key}: {st['members']} members scanned, {st['kept']}/{len(want)} cohort files kept")
        rc = proc.wait()
        if rc != 0:
            raise IOError(f"bzip2 exited with {rc} for {key}")
    except BaseException:
        proc.kill(); proc.wait()
        if os.path.exists(zpath + ".part"):
            os.remove(zpath + ".part")
        raise
    os.replace(zpath + ".part", zpath)
    st["missing"] = sorted(want - seen)
    st["seconds"] = round(time.time() - t0)
    atomic_write(os.path.join(shard_dir, key.replace(".tar.bz2", ".json")), json.dumps(st, indent=1))
    log(f"filtered {key}: {st['members']} members, kept {st['kept']}/{len(want)}, missing {len(st['missing'])}, "
        f"duplicates {len(st['duplicate_members'])}, wrong location {len(st['wrong_location'])}, {st['seconds']}s")
    return st


def fetch_all(a):
    cohort = read_cohort(a.cohort_file)
    arch_dir = os.path.join(a.data_dir, "archives"); shard_dir = os.path.join(a.data_dir, "shards")
    os.makedirs(arch_dir, exist_ok=True); os.makedirs(shard_dir, exist_ok=True)
    archives = list_archives(a.data_dir, a.local_archive, a.archives)
    covered = set(restrict_cohort(cohort, archives))
    if a.archives:
        log(f"archive subset {[ar['key'] for ar in archives]}: {len(covered)} of {len(cohort)} cohort recordings in scope")
    elif not a.local_archive and covered != set(cohort):
        sys.exit(f"{len(set(cohort) - covered)} cohort recordings are not covered by any archive")
    dl = Downloader(conns=a.connections)
    todo = [ar for ar in archives if not os.path.exists(os.path.join(shard_dir, ar["key"].replace(".tar.bz2", ".json")))]
    log(f"{len(archives) - len(todo)} archives already filtered, {len(todo)} to go")
    prefetch = {}

    def start_prefetch(ar):
        dst = os.path.join(arch_dir, ar["key"])
        if ar["local"] or ar["key"] in prefetch or not dl.has_space(dst, ar["size"]):
            return
        th = {"err": None}
        def run():
            try:
                dl.get(ar["key"], ar["size"], ar["md5"], dst)
            except Exception as e:
                th["err"] = e
        th["t"] = threading.Thread(target=run, daemon=True); th["t"].start()
        prefetch[ar["key"]] = th
        log(f"prefetching {ar['key']} in the background")

    for k, ar in enumerate(todo):
        if ar["local"]:
            path = ar["local"]
        else:
            path = os.path.join(arch_dir, ar["key"])
            th = prefetch.pop(ar["key"], None)
            if th:
                th["t"].join()
                if th["err"]:
                    log(f"background download failed ({th['err']}); retrying in the foreground")
            dl.get(ar["key"], ar["size"], ar["md5"], path)
        if k + 1 < len(todo):
            start_prefetch(todo[k + 1])
        split, _ = archive_split_and_prefixes(ar["key"])
        filter_archive(path, ar["key"], cohort_for_archive(cohort, ar["key"]), split, shard_dir)
        if not ar["local"] and not a.no_delete:
            os.remove(path)
            log(f"deleted {ar['key']}; free disk {shutil.disk_usage(a.data_dir).free / GiB:.1f} GiB")
    for th in prefetch.values():
        th["t"].join()


# ----------------------------------------------------------------------------- build
class Bad(Exception):
    def __init__(self, stage, reason):
        super().__init__(reason); self.stage, self.reason = stage, reason


def _bad_constant(name):
    raise Bad("invalid_value", f"JSON contains {name}")


def parse(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Bad("malformed_json", f"not UTF-8: {e}")
    try:
        return json.loads(text, parse_constant=_bad_constant), False
    except Bad:
        raise
    except ValueError:
        pass
    try:  # raw control characters inside strings (seen in some metadata tags); numbers are unaffected
        return json.loads(text, strict=False, parse_constant=_bad_constant), True
    except ValueError as e:
        raise Bad("malformed_json", str(e)[:120])


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten(d):
    """Return {path: ('num', shape, values) | ('cat', value)} for the acoustic sections."""
    out = {}
    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}.{k}")
            return
        if p in EXCLUDED_PATHS:
            return
        if isinstance(o, str):
            out[p] = ("cat", o); return
        if is_num(o):
            out[p] = ("num", (), [o]); return
        if isinstance(o, list) and all(is_num(x) for x in o):
            out[p] = ("num", (len(o),), list(o)); return
        if isinstance(o, list) and o and all(isinstance(r, list) for r in o):
            w = len(o[0])
            if all(len(r) == w and all(is_num(x) for x in r) for r in o):
                out[p] = ("num", (len(o), w), [x for r in o for x in r]); return
        raise Bad("schema", f"unsupported value type at {p}")
    for s in SECTIONS:
        if s not in d or not isinstance(d[s], dict):
            raise Bad("schema", f"missing section {s}")
        walk(d[s], s)
    return out


def signature(flat):
    return tuple(sorted((p, v[0], v[1] if v[0] == "num" else None) for p, v in flat.items()))


def schema_from_signature(sig):
    names, spec = [], []
    for p, kind, shape in sig:
        if kind == "cat":
            if p not in CATEGORICAL:
                raise Bad("schema", f"unexpected text field {p}")
            spec.append((p, "cat", None))
            names += [(f"{p}={lvl}", p.split(".")[0], "onehot") for lvl in CATEGORICAL[p]]
        else:
            spec.append((p, "num", shape))
            if shape == ():
                names.append((p, p.split(".")[0], "numeric"))
            elif len(shape) == 1:
                names += [(f"{p}[{i}]", p.split(".")[0], "numeric") for i in range(shape[0])]
            else:
                names += [(f"{p}[{i},{j}]", p.split(".")[0], "numeric") for i in range(shape[0]) for j in range(shape[1])]
    return spec, names


def vectorise(flat, spec):
    vec = []
    for p, kind, shape in spec:
        v = flat[p]
        if kind == "cat":
            levels = CATEGORICAL[p]
            if v[1] not in levels:
                raise Bad("categorical", f"{p} has unknown value {v[1]!r}")
            vec += [1.0 if v[1] == lvl else 0.0 for lvl in levels]
        else:
            for x in v[2]:
                if not math.isfinite(x) or abs(x) > F32_MAX:
                    raise Bad("invalid_value", f"non-finite or out-of-range value at {p}")
            vec += [float(x) for x in v[2]]
    return vec


def tag(md, key):
    v = (md.get("tags") or {}).get(key)
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    return "" if v is None else str(v)


def build(a):
    import numpy as np
    cohort = read_cohort(a.cohort_file)
    shard_dir = os.path.join(a.data_dir, "shards"); out = os.path.join(a.data_dir, "features")
    os.makedirs(out, exist_ok=True)
    stats = [json.load(open(os.path.join(shard_dir, f))) for f in sorted(os.listdir(shard_dir)) if f.endswith(".json")]
    selected = None
    if a.archives and not a.local_archive:
        selected = list_archives(a.data_dir, None, a.archives)
        cohort = restrict_cohort(cohort, selected)        # recordings in unselected archives are out of scope
        stats = [s for s in stats if s["archive"] in {ar["key"] for ar in selected}]
        log(f"building from archive subset {[ar['key'] for ar in selected]}: {len(cohort)} cohort recordings in scope")
    excl = []          # (mbid, split, subgenre, stage, reason)
    def ex(mbid, stage, reason):
        r = cohort[mbid]; excl.append([mbid, r["split"], r["subgenre"], stage, reason])

    # stage A: which cohort files were found
    found = set()
    filtered_keys = {s["archive"] for s in stats}
    for s in stats:
        for m in s["missing"]:
            ex(m, "missing_file", f"not present in {s['archive']}")
        for m in s["size_mismatch"]:
            ex(m, "malformed_json", f"tar member size mismatch in {s['archive']}")
    dups = {m for s in stats for m in s["duplicate_members"]}
    for m in sorted(dups):
        ex(m, "duplicate", "recording appears more than once in the archives; excluded rather than choosing one")
    expected_archives = set() if a.local_archive else {ar["key"] for ar in (selected or list_archives(a.data_dir, None))}
    unprocessed = expected_archives - filtered_keys
    if unprocessed and not a.allow_partial:
        sys.exit(f"{len(unprocessed)} archives not yet filtered: {sorted(unprocessed)} (use --allow-partial to build anyway)")

    def shard_files():
        for s in stats:
            with zipfile.ZipFile(os.path.join(shard_dir, s["archive"].replace(".tar.bz2", ".zip"))) as z:
                for name in z.namelist():
                    mbid = name.split(".")[0]
                    if mbid not in cohort:
                        sys.exit(f"shard {s['archive']} contains non-cohort file {name}")
                    if mbid in dups or name != f"{mbid}.json":
                        continue
                    yield s, mbid, z.read(name)

    # stage B (pass 1): parse + flatten, keep only each recording's field layout and metadata
    rec_sig, info, sigs = {}, {}, Counter()
    lenient = 0
    for s, mbid, raw in shard_files():
            found.add(mbid)
            try:
                d, was_lenient = parse(raw)
                if not isinstance(d, dict) or "metadata" not in d:
                    raise Bad("schema", "missing metadata section")
                flat = flatten(d)
            except Bad as e:
                ex(mbid, e.stage, e.reason); continue
            lenient += was_lenient
            md = d["metadata"]
            ver = md.get("version") or {}
            tag_id = tag(md, "musicbrainz_recordingid")
            info[mbid] = {
                "artist_mbids": tag(md, "musicbrainz_artistid"),
                "albumartist_mbids": tag(md, "musicbrainz_albumartistid"),
                "tag_recording_mbid": "absent" if not tag_id else ("match" if mbid in tag_id.split("|") else "differs"),
                "essentia_version": ver.get("essentia", ""),
                "essentia_build_sha": ver.get("essentia_build_sha", ""),
                "audio_md5": (md.get("audio_properties") or {}).get("md5_encoded", ""),
                "length_seconds": (md.get("audio_properties") or {}).get("length", ""),
                "lenient_parse": int(was_lenient), "archive": s["archive"]}
            sg = signature(flat); sigs[sg] += 1; rec_sig[mbid] = sg

    # stage C: one fixed schema (frozen to feature_schema.json on first build)
    schema_path = os.path.join(out, "feature_schema.json")
    if os.path.exists(schema_path) and not a.reset_schema:
        canon = tuple(tuple(x[:2]) + ((tuple(x[2]) if x[2] is not None else None),) for x in json.load(open(schema_path)))
    else:
        if not sigs:
            sys.exit("no parsable recordings")
        canon = sigs.most_common(1)[0][0]
        atomic_write(schema_path, json.dumps([list(x) for x in canon]))
    spec, names = schema_from_signature(canon)
    canon_set = set(canon)
    off = sum(n for sg, n in sigs.items() if sg != canon)
    log(f"{len(sigs)} distinct field layouts; {off} recordings differ from the fixed schema")
    if rec_sig and off / len(rec_sig) > 0.01 and not a.allow_schema_loss:
        sys.exit("more than 1% of recordings differ from the fixed schema; inspect before excluding them "
                 "(rerun with --allow-schema-loss to accept)")
    for mbid, sg in rec_sig.items():
        if sg != canon:
            sgs = set(sg)
            miss = sorted(p for p, *_ in canon_set - sgs); extra = sorted(p for p, *_ in sgs - canon_set)
            ex(mbid, "schema", f"missing/changed {miss[:5]} extra {extra[:5]}")

    # stage D (pass 2): vectorise schema-conforming recordings into a pre-sized matrix, in cohort order
    order = [m for m in cohort if rec_sig.get(m) == canon]          # cohort order: split, then recording ID
    pos = {m: i for i, m in enumerate(order)}
    X = np.full((len(order), len(names)), np.nan, dtype=np.float64)
    ok = np.zeros(len(order), dtype=bool)
    for s, mbid, raw in shard_files():
        if mbid not in pos:
            continue
        try:
            vec = vectorise(flatten(parse(raw)[0]), spec)
        except Bad as e:
            ex(mbid, e.stage, e.reason); continue
        if len(vec) != len(names):
            sys.exit(f"internal error: vector length {len(vec)} != {len(names)} for {mbid}")
        if ok[pos[mbid]]:
            sys.exit(f"internal error: {mbid} vectorised twice")
        X[pos[mbid]] = vec; ok[pos[mbid]] = True
    X = X[ok]
    rows = []
    for mbid in (m for m, k in zip(order, ok) if k):
        r = cohort[mbid]
        rows.append({"row": len(rows), "recordingmbid": mbid, "subgenre": r["subgenre"], "split": r["split"],
                     "release_group": r["release_group"], **info[mbid]})

    X32 = X.astype(np.float32)
    if X.size and not np.isfinite(X32).all():
        sys.exit("internal error: non-finite values after float32 cast")
    # duplicate audio (same encoded-file MD5) is flagged, not removed
    md5_groups = Counter(r["audio_md5"] for r in rows if r["audio_md5"])
    for r in rows:
        r["audio_md5_shared"] = int(bool(r["audio_md5"]) and md5_groups[r["audio_md5"]] > 1)
    # constant columns are reported (they carry no information) but kept so the layout stays fixed
    const = [names[j][0] for j in range(len(names)) if X.shape[0] and np.ptp(X[:, j]) == 0]

    np.save(os.path.join(out, "X.npy.tmp.npy"), X32); os.replace(os.path.join(out, "X.npy.tmp.npy"), os.path.join(out, "X.npy"))
    cols = ["row", "recordingmbid", "subgenre", "split", "release_group", "artist_mbids", "albumartist_mbids",
            "tag_recording_mbid", "essentia_version", "essentia_build_sha", "audio_md5", "audio_md5_shared",
            "length_seconds", "lenient_parse", "archive"]
    for fn, hdr, data in [("metadata.csv", cols, [[r[c] for c in cols] for r in rows]),
                          ("feature_names.csv", ["index", "name", "section", "kind"], [[i, *n] for i, n in enumerate(names)]),
                          ("exclusions.csv", ["recordingmbid", "split", "subgenre", "stage", "reason"], sorted(excl, key=lambda e: (e[3], e[0])))]:
        with open(os.path.join(out, fn + ".tmp"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(data)
        os.replace(os.path.join(out, fn + ".tmp"), os.path.join(out, fn))

    # report
    sha = hashlib.sha256(open(os.path.join(out, "X.npy"), "rb").read()).hexdigest()
    kinds = Counter(n[2] for n in names); secs = Counter(n[1] for n in names)
    stage_counts = Counter((e[3], e[1]) for e in excl)
    per = defaultdict(lambda: {sp: [0, 0, set()] for sp in SPLITS})   # cohort, kept, release groups kept
    for m, r in cohort.items():
        per[r["subgenre"]][r["split"]][0] += 1
    for r in rows:
        per[r["subgenre"]][r["split"]][1] += 1; per[r["subgenre"]][r["split"]][2].add(r["release_group"])
    rep = {
        "X_shape": list(X32.shape), "X_dtype": "float32", "X_sha256": sha,
        "features": {"total": len(names), "by_kind": dict(kinds), "by_section": dict(secs),
                     "excluded_paths": sorted(EXCLUDED_PATHS) + ["metadata.* (all)"], "constant_columns": const},
        "archives_selected": sorted(expected_archives) if selected else "all",
        "archives_filtered": len(stats), "archives_unprocessed": sorted(unprocessed),
        "cohort": {sp: sum(1 for r in cohort.values() if r["split"] == sp) for sp in SPLITS},
        "kept": {sp: sum(1 for r in rows if r["split"] == sp) for sp in SPLITS},
        "exclusions_by_stage": {f"{st}/{sp}": n for (st, sp), n in sorted(stage_counts.items())},
        "lenient_parses": lenient, "schema_layouts_seen": len(sigs),
        "tag_recording_mbid": dict(Counter(r["tag_recording_mbid"] for r in rows)),
        "recordings_sharing_audio_md5": sum(r["audio_md5_shared"] for r in rows),
        "essentia_versions": dict(Counter(r["essentia_version"] for r in rows)),
        "per_subgenre": {g: {sp: {"cohort": v[sp][0], "kept": v[sp][1], "release_groups_kept": len(v[sp][2])} for sp in SPLITS}
                         for g, v in sorted(per.items())},
        "members_scanned": sum(s["members"] for s in stats),
        "wrong_location": sum(len(s["wrong_location"]) for s in stats),
    }
    atomic_write(os.path.join(out, "extraction_report.json"), json.dumps(rep, indent=1))
    log(f"build: X {X32.shape}, kept {rep['kept']}, excluded {len(excl)}; report in {out}")
    return rep


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["cohort", "fetch", "build", "run"])
    ap.add_argument("--data-dir", required=True, help="working directory outside iCloud-synced folders")
    ap.add_argument("--tsv-dir", help="Last.fm annotation TSVs (default: <data-dir>/ab_tsv)")
    ap.add_argument("--cohort-file", help="use this cohort CSV instead of <data-dir>/cohort.csv (testing)")
    ap.add_argument("--local-archive", action="append", help="process a local .tar.bz2 instead of Zenodo (testing)")
    ap.add_argument("--archives", help="comma-separated archives to download and process, e.g. "
                    "train-89,validation-01234567 (default: all 10)")
    ap.add_argument("--connections", type=int, default=8)
    ap.add_argument("--no-delete", action="store_true", help="keep archives after filtering")
    ap.add_argument("--allow-partial", action="store_true", help="build even if some archives are not filtered yet")
    ap.add_argument("--allow-schema-loss", action="store_true")
    ap.add_argument("--reset-schema", action="store_true", help="re-derive feature_schema.json")
    a = ap.parse_args()
    a.data_dir = os.path.abspath(os.path.expanduser(a.data_dir))
    a.archives = [x.strip() for x in a.archives.split(",") if x.strip()] if a.archives else None
    os.makedirs(a.data_dir, exist_ok=True)
    tsv = a.tsv_dir or os.path.join(a.data_dir, "ab_tsv")
    if a.step in ("cohort", "run") and not a.cohort_file:
        a.cohort_file = build_cohort(a.data_dir, tsv)
    a.cohort_file = a.cohort_file or os.path.join(a.data_dir, "cohort.csv")
    if a.step in ("fetch", "run"):
        fetch_all(a)
    if a.step in ("build", "run"):
        build(a)


if __name__ == "__main__":
    main()
