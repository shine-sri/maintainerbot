import os, sys, json, time, requests
from pathlib import Path

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
HEAD = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEAD["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
MAX_PAGES = int(os.getenv("MAX_PAGES", "15"))  # 15 pages = up to 1500 issues/repo

def get(url, params):
    while True:
        r = requests.get(url, headers=HEAD, params=params, timeout=30)
        if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = int(r.headers["X-RateLimit-Reset"]) - time.time() + 5
            print(f"rate limited, sleeping {wait:.0f}s"); time.sleep(max(wait, 5)); continue
        r.raise_for_status()
        return r

def download(repo):
    out = RAW / (repo.replace("/", "__") + ".jsonl")
    if out.exists():
        print("cached:", repo); return
    rows = []
    for page in range(1, MAX_PAGES + 1):
        items = get(f"https://api.github.com/repos/{repo}/issues",
                    {"state": "closed", "per_page": 100, "page": page}).json()
        if not items: break
        for it in items:
            if "pull_request" in it: continue
            rows.append({"repo": repo, "number": it["number"], "title": it["title"],
                         "body": it.get("body") or "",
                         "labels": [l["name"] for l in it["labels"]]})
        print(repo, "page", page, "issues", len(rows))
    out.write_text("\n".join(json.dumps(r) for r in rows))  # written only when complete

if __name__ == "__main__":
    for repo in sys.argv[1:]:
        download(repo)