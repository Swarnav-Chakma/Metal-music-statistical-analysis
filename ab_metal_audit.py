#!/usr/bin/env python3
"""
Feasibility audit for metal subgenres in the AcousticBrainz Genre Dataset (MediaEval).

Uses ONLY the public annotation TSVs (Discogs, Last.fm, Tagtraum; train + validation),
about 100 MB compressed. It does NOT touch the feature archives.

Usage
  python ab_metal_audit.py --download --data-dir ab_tsv --out audit_out
  python ab_metal_audit.py --data-dir ab_tsv --out audit_out      (files already present)

Outputs (in --out)
  labels_<source>.csv         every metal-related label: role, family, recordings, release groups, per split
  unclassified_labels.csv     metal-looking labels not in the role table (review these by hand)
  overlap_all.csv             pairwise recording intersections, all genres
  overlap_metal.csv           pairwise intersections restricted to metal (both / only one side)
  compat_<a>_<b>.csv          per shared metal recording: compatibility category and the labels behind it
  compat_summary.csv          category counts per pair (recordings and distinct release groups)
  lastfm_single_label.csv     Last.fm recordings with exactly one genuine metal subgenre, per label
  summary.md                  human-readable report of all of the above
Standard library only.
"""
import argparse, bz2, csv, itertools, os, re, sys, urllib.request
from collections import defaultdict, Counter

ZENODO = "https://zenodo.org/records/2553414/files/{}?download=1"
SOURCES = ["discogs", "lastfm", "tagtraum"]
SPLITS = ["train", "validation"]
FNAME = "acousticbrainz-mediaeval-{s}-{p}.tsv.bz2"

# ---------------------------------------------------------------------------
# Role table. Keys are normalised labels (see norm()). Roles:
#   P parent/umbrella, S genuine subgenre, H hybrid subgenre (genuine),
#   R regional/era/scene descriptor, V vague descriptor, A metal-adjacent (not metal)
# Families: canonical style buckets used for cross-source comparison.
#   Hybrids map to several families; refinements map to their base family
#   (e.g. technical death metal -> death) so granularity is never read as conflict.
# ---------------------------------------------------------------------------
R = {}
def add(src, label, role, fams=()):
    R[(src, label)] = (role, frozenset(fams))

# Discogs (metal styles sit under rock; no metal parent exists)
for lab, f in [("heavymetal", ["heavy"]), ("deathmetal", ["death"]), ("blackmetal", ["black"]),
               ("thrash", ["thrash"]), ("doommetal", ["doom"]), ("grindcore", ["grind"]),
               ("numetal", ["nu"]), ("speedmetal", ["speed"]), ("powermetal", ["power"]),
               ("vikingmetal", ["viking"]), ("metalcore", ["metalcore"]), ("funkmetal", ["funk"]),
               ("gothicmetal", ["gothic"]), ("folkmetal", ["folk"]), ("progressivemetal", ["prog"]),
               ("sludgemetal", ["sludge"]), ("melodicdeathmetal", ["melodeath", "death"])]:
    add("discogs", "rock---" + lab, "S", f)
for lab in ["stonerrock", "hardcore", "crust", "gothrock", "industrial", "posthardcore", "melodichardcore", "deathrock"]:
    add("discogs", "rock---" + lab, "A")

# Last.fm
add("lastfm", "metal", "P")
for lab, f in [("heavymetal", ["heavy"]), ("deathmetal", ["death"]), ("blackmetal", ["black"]),
               ("thrashmetal", ["thrash"]), ("progressivemetal", ["prog"]), ("powermetal", ["power"]),
               ("doommetal", ["doom"]), ("gothicmetal", ["gothic"]), ("metalcore", ["metalcore"]),
               ("melodicdeathmetal", ["melodeath", "death"]), ("symphonicmetal", ["symphonic"]),
               ("folkmetal", ["folk"]), ("grindcore", ["grind"]), ("brutaldeathmetal", ["death"]),
               ("speedmetal", ["speed"]), ("vikingmetal", ["viking"]), ("technicaldeathmetal", ["death"]),
               ("deathcore", ["deathcore", "death", "metalcore"]), ("groovemetal", ["groove"]),
               ("sludgemetal", ["sludge"]), ("paganmetal", ["pagan"]), ("avantgardemetal", ["avant"]),
               ("mathcore", ["mathcore"])]:
    add("lastfm", "metal---" + lab, "S", f)
for lab, f in [("atmosphericblackmetal", ["black"]), ("melodicblackmetal", ["black"]),
               ("symphonicblackmetal", ["black", "symphonic"]), ("progressivedeathmetal", ["prog", "death"]),
               ("deathdoommetal", ["death", "doom"]), ("melodicmetalcore", ["metalcore"]),
               ("blackeneddeathmetal", ["black", "death"])]:
    add("lastfm", "metal---" + lab, "H", f)
for lab, f in [("oldschooldeathmetal", ["death"]), ("norwegianblackmetal", ["black"]),
               ("classicmetal", []), ("norwegian", []), ("finnishmetal", []), ("classicbritishheavymetal", ["heavy"]),
               ("germanmetal", []), ("britishmetal", []), ("swedishmetal", [])]:
    add("lastfm", "metal---" + lab, "R", f)          # regional/era: family kept only where unambiguous
for lab in ["melodicmetal", "darkmetal", "epicmetal"]:
    add("lastfm", "metal---" + lab, "V")
for lab, f in [("electronic---industrialmetal", ["industrial"]), ("rock---alternativemetal", ["alternative"]),
               ("rock---funkmetal", ["funk"]), ("rock---hairmetal", ["glam"]), ("rock---numetal", ["nu"]),
               ("rock---stonermetal", ["stoner"]), ("postrock---postmetal", ["post"])]:
    add("lastfm", lab, "S", f)
for lab in ["rock---rapcore", "rock---stoner", "rock---stonerrock", "posthardcore", "posthardcore---emocore",
            "posthardcore---screamo", "rock---hardcorepunk", "gothic---gothicrock", "rock---powerballad"]:
    add("lastfm", lab, "A")

# Tagtraum (metal is a subgenre-level label under rock/pop; treated as parent)
add("tagtraum", "rock/pop---metal", "P")
add("tagtraum", "rock/pop---extrememetal", "P")
for lab, f in [("heavymetal", ["heavy"]), ("deathmetal", ["death"]), ("thrashmetal", ["thrash"]),
               ("progressivemetal", ["prog"]), ("metalcore", ["metalcore"]), ("powermetal", ["power"]),
               ("alternativemetal", ["alternative"]), ("gothicmetal", ["gothic"]), ("symphonicmetal", ["symphonic"]),
               ("doommetal", ["doom"]), ("melodicdeathmetal", ["melodeath", "death"]), ("folkmetal", ["folk"]),
               ("numetal", ["nu"]), ("deathcore", ["deathcore", "death", "metalcore"]), ("popmetal", ["glam"]),
               ("funkmetal", ["funk"])]:
    add("tagtraum", "rock/pop---" + lab, "S", f)
add("tagtraum", "industrial---industrialmetal", "S", ["industrial"])
for lab in ["rock/pop---stonerrock", "rock/pop---posthardcore", "rock/pop---hardcorepunk", "rock/pop---gothicrock"]:
    add("tagtraum", lab, "A")

# Non-metal labels that the loose METALISH regex catches (verified against the official vocabulary)
for _src, _labs in [("discogs", ["electronic---breakcore", "electronic---happyhardcore", "electronic---hardcore",
                                  "electronic---speedcore", "hiphop---hardcorehiphop", "hiphop---horrorcore", "stage&screen---score"]),
                     ("lastfm", ["electronic---breakcore", "electronic---happyhardcore", "soundtrack---score"]),
                     ("tagtraum", ["electronic---breakcore", "soundtrack---score"])]:
    for _l in _labs:
        add(_src, _l, "X")

METAL_ROLES = set("PSHRV")      # A is excluded from "metal"
GENUINE = set("SH")
# Pre-declared neighbouring families: disjoint but closely related styles, reported as "adjacent"
NEIGHBOURS = {frozenset(p) for p in [("thrash", "speed"), ("heavy", "speed"), ("heavy", "power"),
              ("doom", "sludge"), ("folk", "viking"), ("folk", "pagan"), ("viking", "pagan"),
              ("gothic", "doom"), ("metalcore", "deathcore"), ("nu", "alternative"), ("power", "symphonic")]}
UMBRELLA = {"heavy"}   # "heavy metal" is used both as a style and as a catch-all
METALISH = re.compile(r"metal|core$|grind|doom|sludge|djent|thrash|stoner|viking|pagan|nwobhm")

def norm(label):
    parts = label.strip().lower().split("---")
    return "---".join(re.sub(r"[\s\-]", "", p) for p in parts)

def families_in_taxonomy(src):
    out = set()
    for (s, _), (role, fams) in R.items():
        if s == src and role in GENUINE | {"R"}:
            out |= fams
    return out

# ---------------------------------------------------------------------------
def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for s in SOURCES:
        for p in SPLITS:
            fn = FNAME.format(s=s, p=p)
            dst = os.path.join(data_dir, fn)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue
            print("downloading", fn, file=sys.stderr)
            urllib.request.urlretrieve(ZENODO.format(fn), dst + ".part")
            os.replace(dst + ".part", dst)

def load(data_dir):
    """data[src][rec] = (rg, split, set(norm labels)); raw[src] = Counter of raw label strings"""
    data, raw = {}, {}
    for s in SOURCES:
        d, rc = {}, Counter()
        for p in SPLITS:
            path = os.path.join(data_dir, FNAME.format(s=s, p=p))
            if not os.path.exists(path):
                print(f"WARNING missing {path}", file=sys.stderr); continue
            with bz2.open(path, "rt", encoding="utf-8", newline="") as f:
                r = csv.reader(f, delimiter="\t")
                header = next(r)
                if not header or header[0] != "recordingmbid":
                    sys.exit(f"unexpected header in {path}: {header[:3]}")
                for row in r:
                    if len(row) < 2 or not row[0]:
                        continue
                    labs = [x for x in row[2:] if x.strip()]
                    rc.update(labs)
                    if row[0] in d:
                        print(f"WARNING duplicate recording {row[0]} in {s}", file=sys.stderr)
                    d[row[0]] = (row[1], p, {norm(x) for x in labs})
        data[s], raw[s] = d, rc
    return data, raw

def metal_labels(src, labs):
    return {l: R[(src, l)] for l in labs if (src, l) in R and R[(src, l)][0] in METAL_ROLES}

def is_metal(src, labs):
    return bool(metal_labels(src, labs))

def compare(a, la, b, lb, comparable):
    """Return (category, detail). Categories: compatible, adjacent, conflicting,
    insufficient (no comparable family on a side), umbrella (only 'heavy' on a side vs specific other)."""
    fa = set().union(*[f for (_, f) in metal_labels(a, la).values()]) & comparable if la else set()
    fb = set().union(*[f for (_, f) in metal_labels(b, lb).values()]) & comparable if lb else set()
    if not fa or not fb:
        return "insufficient", (sorted(fa), sorted(fb))
    if fa & fb:
        return "compatible", (sorted(fa), sorted(fb))
    if fa <= UMBRELLA or fb <= UMBRELLA:
        return "umbrella", (sorted(fa), sorted(fb))
    if any(frozenset((x, y)) in NEIGHBOURS for x in fa for y in fb):
        return "adjacent", (sorted(fa), sorted(fb))
    return "conflicting", (sorted(fa), sorted(fb))

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--features", type=int, default=2675,
                    help="fixed numeric attributes after one-hot (2,675 from a sample JSON; paper says 2,669)")
    ap.add_argument("--min-rg", type=int, default=100)
    a = ap.parse_args()
    if a.download:
        download(a.data_dir)
    os.makedirs(a.out, exist_ok=True)
    data, raw = load(a.data_dir)
    md = ["# AcousticBrainz metal audit\n"]

    # 1. labels per source -------------------------------------------------
    unclassified = []
    for s in SOURCES:
        agg = defaultdict(lambda: {sp: [0, set()] for sp in SPLITS})
        for rec, (rg, sp, labs) in data[s].items():
            for l in labs:
                if (s, l) in R or METALISH.search(l.split("---")[-1]):
                    agg[l][sp][0] += 1; agg[l][sp][1].add(rg)
        rows = []
        for l, v in agg.items():
            role, fams = R.get((s, l), ("?", frozenset()))
            if role == "?":
                unclassified.append((s, l, v["train"][0] + v["validation"][0]))
            rec_all = v["train"][0] + v["validation"][0]
            rg_all = len(v["train"][1] | v["validation"][1])
            rows.append([l, role, "|".join(sorted(fams)), v["train"][0], len(v["train"][1]),
                         v["validation"][0], len(v["validation"][1]), rec_all, rg_all])
        rows.sort(key=lambda r: (r[1], -r[7]))
        with open(os.path.join(a.out, f"labels_{s}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["label", "role", "families", "rec_train", "rg_train",
                                           "rec_val", "rg_val", "rec_all", "rg_all"]); w.writerows(rows)
        gen = [r for r in rows if r[1] in GENUINE]
        n_metal = sum(1 for _, _, labs in data[s].values() if is_metal(s, labs))
        md.append(f"## {s}: {len(data[s])} recordings; {n_metal} with any metal label (roles P/S/H/R/V)\n")
        md.append(f"genuine subgenres present: {len(gen)}; with >= {a.min_rg} release groups (train+val): "
                  f"{sum(1 for r in gen if r[8] >= a.min_rg)}\n")
        md.append("| label | role | rec (train/val/all) | RG (all) |\n|---|---|---|---|")
        for r in rows:
            if r[1] in METAL_ROLES:
                md.append(f"| {r[0]} | {r[1]} | {r[3]}/{r[5]}/{r[7]} | {r[8]} |")
        md.append("")
    with open(os.path.join(a.out, "unclassified_labels.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "label", "recordings"]); w.writerows(sorted(unclassified))
    md.append(f"Unclassified metal-looking labels: {len(unclassified)} (see unclassified_labels.csv; "
              "if non-zero, classify them before trusting the counts)\n")
    md.append("Raw label spelling check (first 3 raw strings per source): " +
              "; ".join(f"{s}: {[k for k in raw[s] if 'metal' in k.lower()][:3]}" for s in SOURCES) + "\n")

    # 1b. composition and thresholds (A3, A4) --------------------------------------
    md.append("## Recording composition and thresholds\n")
    md.append("| source | recordings with any metal label | exactly 1 genuine subgenre | 2+ genuine | "
              "descriptor only (R/V, no S/H) | broad parent only |\n|---|---|---|---|---|---|")
    comp_rows = []
    for s in SOURCES:
        c = Counter()
        for rec, (rg, sp, labs) in data[s].items():
            ml = metal_labels(s, labs)
            if not ml:
                continue
            roles = Counter(r for (r, _) in ml.values())
            g = roles["S"] + roles["H"]
            c["any"] += 1
            if g == 1: c["one"] += 1
            elif g >= 2: c["multi"] += 1
            elif roles["R"] + roles["V"]: c["desc"] += 1
            else: c["parent"] += 1
        comp_rows.append([s, c["any"], c["one"], c["multi"], c["desc"], c["parent"]])
        md.append(f"| {s} | {c['any']} | {c['one']} | {c['multi']} | {c['desc']} | {c['parent']} |")
    with open(os.path.join(a.out, "composition.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "any_metal", "exactly_one_genuine", "two_plus_genuine",
                                       "descriptor_only", "parent_only"]); w.writerows(comp_rows)
    md.append("\nGenuine-subgenre counts passing each threshold (train+validation). "
              "'all labels' counts every recording carrying the label; 'single-label' counts only recordings "
              "whose sole genuine subgenre it is.\n")
    md.append("| source | scope | labels | >=100 rec | >=500 rec | >=100 RG | all three |\n|---|---|---|---|---|---|---|")
    thr_rows = []
    for s in SOURCES:
        allc = defaultdict(lambda: [0, set()]); single = defaultdict(lambda: [0, set()])
        for rec, (rg, sp, labs) in data[s].items():
            g = [l for l in labs if R.get((s, l), ("?",))[0] in GENUINE]
            for l in g:
                allc[l][0] += 1; allc[l][1].add(rg)
            if len(g) == 1:
                single[g[0]][0] += 1; single[g[0]][1].add(rg)
        scopes = [("all labels", allc), ("single-label", single)]
        if s == "lastfm":
            scopes += [("all labels, metal--- parent only", {k: v for k, v in allc.items() if k.startswith("metal---")}),
                       ("single-label, metal--- parent only", {k: v for k, v in single.items() if k.startswith("metal---")})]
        for name, d in scopes:
            n100 = sum(1 for v in d.values() if v[0] >= 100); n500 = sum(1 for v in d.values() if v[0] >= 500)
            g100 = sum(1 for v in d.values() if len(v[1]) >= 100)
            both = sum(1 for v in d.values() if v[0] >= 500 and len(v[1]) >= 100)
            md.append(f"| {s} | {name} | {len(d)} | {n100} | {n500} | {g100} | {both} |")
            thr_rows.append([s, name, len(d), n100, n500, g100, both])
    with open(os.path.join(a.out, "thresholds.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "scope", "labels", "ge100_rec", "ge500_rec", "ge100_rg",
                                       "ge500rec_and_ge100rg"]); w.writerows(thr_rows)

    # 1c. leakage checks (C5) ----------------------------------------------------------
    md.append("\n## Leakage checks\n")
    for s in SOURCES:
        rg_split = defaultdict(set)
        for rec, (rg, sp, labs) in data[s].items():
            rg_split[rg].add(sp)
        cross = sum(1 for v in rg_split.values() if len(v) > 1)
        md.append(f"- {s}: {len(rg_split)} release groups; {cross} appear in both train and validation")
    for x, y in itertools.combinations(SOURCES, 2):
        shared = data[x].keys() & data[y].keys()
        diff = sum(1 for r in shared if data[x][r][0] != data[y][r][0])
        spl = sum(1 for r in shared if data[x][r][1] != data[y][r][1])
        md.append(f"- {x} & {y}: {diff} shared recordings with a different release group; "
                  f"{spl} shared recordings in different splits")
    md.append("")

    # 2. overlaps ---------------------------------------------------------------
    md.append("## Overlaps\n| pair | shared (all genres) | both metal | metal in first only | metal in second only |\n|---|---|---|---|---|")
    ov_all, ov_metal = [], []
    for x, y in itertools.combinations(SOURCES, 2):
        shared = data[x].keys() & data[y].keys()
        both = [r for r in shared if is_metal(x, data[x][r][2]) and is_metal(y, data[y][r][2])]
        xo = sum(1 for r in shared if is_metal(x, data[x][r][2]) and not is_metal(y, data[y][r][2]))
        yo = sum(1 for r in shared if is_metal(y, data[y][r][2]) and not is_metal(x, data[x][r][2]))
        ov_all.append([x, y, len(shared)]); ov_metal.append([x, y, len(both), xo, yo])
        md.append(f"| {x} & {y} | {len(shared)} | {len(both)} | {xo} | {yo} |")
    tri = data["discogs"].keys() & data["lastfm"].keys() & data["tagtraum"].keys()
    tri_m = [r for r in tri if all(is_metal(s, data[s][r][2]) for s in SOURCES)]
    md.append(f"\nAll three public sources: {len(tri)} shared; {len(tri_m)} metal in all three\n")
    for fn, rows, hdr in [("overlap_all.csv", ov_all, ["a", "b", "shared"]),
                          ("overlap_metal.csv", ov_metal, ["a", "b", "both_metal", "a_only", "b_only"])]:
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

    # 3. compatibility ------------------------------------------------------------
    md.append("## Compatibility of subgenre annotations on shared metal recordings\n")
    md.append("Families compared only where both taxonomies define them; refinements map to base families.\n")
    md.append("| pair | comparable families | compatible | adjacent | umbrella | conflicting | insufficient |\n|---|---|---|---|---|---|---|")
    summ = []
    for x, y in itertools.combinations(SOURCES, 2):
        comp = families_in_taxonomy(x) & families_in_taxonomy(y)
        cnt, rgs = Counter(), defaultdict(set)
        with open(os.path.join(a.out, f"compat_{x}_{y}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["recordingmbid", f"rg_{x}", f"rg_{y}", "category",
                                           f"families_{x}", f"families_{y}", f"labels_{x}", f"labels_{y}"])
            for r in sorted(data[x].keys() & data[y].keys()):
                lx, ly = data[x][r][2], data[y][r][2]
                if not (is_metal(x, lx) and is_metal(y, ly)):
                    continue
                cat, (fx, fy) = compare(x, lx, y, ly, comp)
                cnt[cat] += 1; rgs[cat].add(data[x][r][0])
                w.writerow([r, data[x][r][0], data[y][r][0], cat, "|".join(fx), "|".join(fy),
                            "|".join(sorted(metal_labels(x, lx))), "|".join(sorted(metal_labels(y, ly)))])
        cats = ["compatible", "adjacent", "umbrella", "conflicting", "insufficient"]
        md.append(f"| {x} & {y} | {len(comp)} | " + " | ".join(f"{cnt[c]} ({len(rgs[c])} RG)" for c in cats) + " |")
        summ.append([x, y, len(comp)] + [cnt[c] for c in cats] + [len(rgs[c]) for c in cats])
    with open(os.path.join(a.out, "compat_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "n_comparable_families", "compatible", "adjacent", "umbrella", "conflicting",
                    "insufficient", "rg_compatible", "rg_adjacent", "rg_umbrella", "rg_conflicting", "rg_insufficient"])
        w.writerows(summ)

    # 4. Last.fm single-label analytic subset ---------------------------------------
    per = defaultdict(lambda: [0, set()])
    for rec, (rg, sp, labs) in data["lastfm"].items():
        g = [l for l in labs if R.get(("lastfm", l), ("?",))[0] in GENUINE]
        if len(g) == 1:
            per[g[0]][0] += 1; per[g[0]][1].add(rg)
    rows = sorted(([l, n, len(s)] for l, (n, s) in per.items()), key=lambda r: -r[1])
    with open(os.path.join(a.out, "lastfm_single_label.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "recordings", "release_groups"]); w.writerows(rows)
    ok = [r for r in rows if r[2] >= a.min_rg]
    md.append(f"\n## Last.fm single-subgenre subset\n{sum(r[1] for r in rows)} recordings carry exactly one genuine "
              f"metal subgenre; {len(ok)} labels have >= {a.min_rg} release groups.\n")
    md.append("| label | recordings | release groups |\n|---|---|---|")
    md += [f"| {l} | {n} | {g} |" for l, n, g in rows]

    # 5. size arithmetic ---------------------------------------------------------------
    nL = sum(1 for _, _, labs in data["lastfm"].values() if is_metal("lastfm", labs))
    nDL = ov_metal[[i for i, r in enumerate(ov_metal) if r[:2] == ["discogs", "lastfm"]][0]][2]
    md.append(f"\n## Size check (features = {a.features} per recording, labels and derived columns excluded)\n")
    for name, n in [("Last.fm metal recordings", nL), ("Discogs & Last.fm shared metal recordings", nDL)]:
        md.append(f"- {name}: {n} x {a.features} = {n * a.features:,} -> "
                  f"{'meets' if n * a.features >= 3_000_000 else 'below'} 3,000,000")
    md.append(f"- minimum instances needed: ceil(3,000,000 / {a.features}) = {-(-3_000_000 // a.features)}")
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
