import json, os
import audit
import github_api as gh
from classifier import Classifier
from retrieval import DocsIndex
from gemma import write_reply

LIVE = os.getenv("MB_LIVE", "false").lower() == "true"      # dry-run unless exactly "true"
CLS_T = float(os.getenv("MB_CLS_THRESHOLD", "0.8"))
DOC_T = float(os.getenv("MB_DOC_THRESHOLD", "0.70"))         # tuned from try_retrieval scores
DOCS = os.getenv("MB_DOCS_PATH", "README.md")
PREFIX = "mb:"
MARKER = "<!-- maintainerbot -->"


def main():
    event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
    issue = event["issue"]
    num, title, body = issue["number"], issue["title"], issue.get("body") or ""

    if "pull_request" in issue or issue["user"]["type"] == "Bot":
        audit.log(issue=num, action="none", reason="PR or bot author", live=LIVE)
        return

    label, conf = Classifier().predict(f"{title}\n\n{body}")
    rec = {"issue": num, "label": label, "confidence": round(conf, 3), "live": LIVE}

    if conf < CLS_T:
        audit.log(**rec, action="none", reason=f"abstained: confidence below {CLS_T}")
        return

    actions = [f"label:{PREFIX}{label}"]
    reply = None
    if label == "question":
        chunk, score = DocsIndex(open(DOCS).read()).search(f"{title}\n{body[:500]}")
        rec["retrieval_score"] = round(score, 3)
        rec["section"] = chunk["heading"]
        if score >= DOC_T:
            text, source = write_reply(title, chunk)
            rec["reply_source"] = source
            if text:
                reply = (f"{MARKER}\n{text}\n\n"
                         f"_Automated reply from MaintainerBot (section: {chunk['heading']})._")
                actions.append("comment:docs-answer")
                rec["reply_preview"] = text[:300]
        if not reply:
            rec["note"] = "not documented: left to a human"

    if LIVE:
        gh.add_labels(num, [f"{PREFIX}{label}"])
        if reply:
            gh.comment(num, reply)
    audit.log(**rec, action=actions,
              reason="confident prediction" + ("" if LIVE else " (dry-run: nothing posted)"))


if __name__ == "__main__":
    main()