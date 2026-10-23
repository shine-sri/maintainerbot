import os, requests
from urllib.parse import quote

API = "https://api.github.com"
REPO = os.environ.get("GITHUB_REPOSITORY", "")


def _h():
    return {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json"}


def add_labels(num, labels):
    requests.post(f"{API}/repos/{REPO}/issues/{num}/labels", headers=_h(),
                  json={"labels": labels}, timeout=30).raise_for_status()


def comment(num, body):
    requests.post(f"{API}/repos/{REPO}/issues/{num}/comments", headers=_h(),
                  json={"body": body}, timeout=30).raise_for_status()


def remove_label(num, label):
    r = requests.delete(f"{API}/repos/{REPO}/issues/{num}/labels/{quote(label, safe='')}",
                        headers=_h(), timeout=30)
    if r.status_code != 404:
        r.raise_for_status()


def delete_label(label):
    requests.delete(f"{API}/repos/{REPO}/labels/{quote(label, safe='')}",
                    headers=_h(), timeout=30)


def delete_comment(cid):
    requests.delete(f"{API}/repos/{REPO}/issues/comments/{cid}",
                    headers=_h(), timeout=30).raise_for_status()


def paged(url, params=None):
    p = dict(params or {})
    p["per_page"] = 100
    page = 1
    while True:
        p["page"] = page
        r = requests.get(url, headers=_h(), params=p, timeout=30)
        r.raise_for_status()
        items = r.json()
        if not items:
            return
        yield from items
        page += 1