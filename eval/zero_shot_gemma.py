import time, sys, argparse, random
import ollama
sys.path.insert(0, "eval")
from common import load, report

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma3:1b")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()

LABELS = ["bug", "feature", "question", "docs"]
PROMPT = ("You are a GitHub issue classifier. Answer with exactly one word from: "
          "bug, feature, question, docs. No explanation.\n\n"
          "Issue:\n{text}\n\nAnswer:")


def predict(text):
    out = ollama.chat(model=a.model,
                      messages=[{"role": "user", "content": PROMPT.format(text=text[:1200])}],
                      options={"temperature": 0, "num_predict": 20})["message"]["content"].lower()
    out = out.replace("documentation", "docs").replace("enhancement", "feature")
    hits = [(out.find(l), l) for l in LABELS if l in out]
    return min(hits)[1] if hits else "bug"  # earliest label wins; unparseable -> bug


test = load("test")
random.Random(0).shuffle(test)  # test file is grouped by source file, so shuffle before sampling
test = test[: a.n]
y = [r["label"] for r in test]

t0 = time.time()
p = [predict(r["text"]) for r in test]
report(f"Zero-shot {a.model} (n={a.n})", y, p, time.time() - t0)