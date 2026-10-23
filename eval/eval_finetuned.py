import sys, json, time
sys.path.insert(0, "eval"); sys.path.insert(0, "bot")
from common import load, report
from classifier import Classifier
from sklearn.metrics import accuracy_score, f1_score

test = load("test")
clf = Classifier()
y = [r["label"] for r in test]
t0 = time.time()
out = [clf.predict(r["text"]) for r in test]
secs = time.time() - t0
pred = [o[0] for o in out]; conf = [o[1] for o in out]
report("Fine-tuned DistilBERT (CPU)", y, pred, secs)

with open("eval/test_preds.jsonl", "w") as f:
    for r, p, c in zip(test, pred, conf):
        f.write(json.dumps({"repo": r["repo"], "number": r["number"], "true": r["label"],
                            "pred": p, "conf": round(c, 4)}) + "\n")

print("\nthreshold  coverage  acc_on_answered  macro_f1_on_answered")
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    idx = [i for i, c in enumerate(conf) if c >= t]
    if not idx: continue
    yt = [y[i] for i in idx]; yp = [pred[i] for i in idx]
    print(f"{t:<10} {len(idx)/len(y):<9.2f} {accuracy_score(yt, yp):<16.3f} {f1_score(yt, yp, average='macro'):.3f}")