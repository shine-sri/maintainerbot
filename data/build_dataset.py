import json, argparse, collections, random
from pathlib import Path

MAP = {
    "bug": {"bug", "kind/bug", "type: bug", "type/bug", "defect", "bug report", "00 - bug"},
    "feature": {"enhancement", "feature", "kind/feature", "kind/enhancement", "feature request",
                "new feature", "type: feature", "type/enhancement", "type/feature", "01 - enhancement"},
    "question": {"question", "kind/question", "usage question", "support", "type: question",
                 "type/question", "question/not a bug", "33 - question"},
    "docs": {"documentation", "docs", "kind/documentation", "kind/docs", "area/docs",
             "type: documentation", "type/docs", "04 - documentation"},
}
LOOKUP = {name: cls for cls, names in MAP.items() for name in names}


def classify(labels):
    classes = {LOOKUP[l.lower()] for l in labels if l.lower() in LOOKUP}
    return classes.pop() if len(classes) == 1 else None  # drop none / ambiguous


ap = argparse.ArgumentParser()
ap.add_argument("--test", nargs="+", required=True, help="repos held out for test, e.g. numpy__numpy")
ap.add_argument("--val", nargs="*", default=[], help="repos held out for validation (optional)")
ap.add_argument("--val-frac", type=float, default=0.0,
                help="stratified fraction of train moved to val, e.g. 0.15")
a = ap.parse_args()

splits = collections.defaultdict(list)
seen = set()
for f in sorted(Path("data/raw").glob("*.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        key = (r["repo"], r["number"])
        if key in seen:
            continue
        seen.add(key)
        label = classify(r["labels"])
        if not label:
            continue
        repo_key = r["repo"].replace("/", "__")
        split = "test" if repo_key in a.test else "val" if repo_key in a.val else "train"
        text = (r["title"] + "\n\n" + r["body"])[:1500]
        splits[split].append({"repo": r["repo"], "number": r["number"], "text": text, "label": label})

if a.val_frac > 0:
    rng = random.Random(0)
    by_label = collections.defaultdict(list)
    for r in splits["train"]:
        by_label[r["label"]].append(r)
    train, val = [], list(splits["val"])
    for rows in by_label.values():
        rng.shuffle(rows)
        k = int(len(rows) * a.val_frac)
        val += rows[:k]
        train += rows[k:]
    splits["train"], splits["val"] = train, val

for s in ("train", "val", "test"):
    rows = splits[s]
    Path(f"data/{s}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(s, len(rows), dict(collections.Counter(r["label"] for r in rows)),
          "repos:", sorted({r["repo"] for r in rows}))