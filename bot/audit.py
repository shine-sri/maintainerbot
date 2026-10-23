import json, os, time

LOG = os.getenv("MB_AUDIT_LOG", "audit/audit.jsonl")


def log(**kw):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")