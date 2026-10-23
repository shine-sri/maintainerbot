import os, json, time, requests
from pathlib import Path

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
HEAD = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEAD["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))  # 5 pages = up to 500 issues per (repo, label)

# (repo, label) pairs; each label filter gives a class-balanced sample
JOBS = [
    ("pandas-dev/pandas", "Bug"), ("pandas-dev/pandas", "Enhancement"),
    ("pandas-dev/pandas", "Docs"), ("pandas-dev/pandas", "Usage Question"),
    ("scikit-learn/scikit-learn", "Bug"), ("scikit-learn/scikit-learn", "New Feature"),
    ("scikit-learn/scikit-learn", "Documentation"), ("scikit-learn/scikit-learn", "Question"),
    ("numpy/numpy", "00 - Bug"), ("numpy/numpy", "01 - Enhancement"),
    ("numpy/numpy", "04 - Documentation"), ("numpy/numpy", "33 - Question"),
]


def get(url, params):
    while True:
        r = requests.get(url, headers=HEAD, params=params, timeout=30)
        if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = int(r.headers["X-RateLimit-Reset"]) - time.time() + 5
            print(f"rate limited, sleeping {wait:.0f}s")
            time.sleep(max(wait, 5))
            continue
        if r.status_code == 422:  # GitHub pagination cap
            print("pagination cap reached, stopping this job")
            return []
        r.raise_for_status()
        return r.json()


for repo, label in JOBS:
    slug = label.lower().replace(" ", "_").replace("-", "")
    out = RAW / f"{repo.replace('/', '__')}__{slug}.jsonl"
    if out.exists():
        print("cached:", repo, label)
        continue
    rows = []
    for page in range(1, MAX_PAGES + 1):
        items = get(f"https://api.github.com/repos/{repo}/issues",
                    {"state": "closed", "labels": label, "per_page": 100, "page": page})
        if not items:
            break
        for it in items:
            if "pull_request" in it:
                continue
            rows.append({"repo": repo, "number": it["number"], "title": it["title"],
                         "body": it.get("body") or "",
                         "labels": [l["name"] for l in it["labels"]]})
        print(repo, label, "page", page, "issues so far", len(rows))
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print("saved", out, len(rows))