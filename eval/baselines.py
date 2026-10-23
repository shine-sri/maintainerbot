import time, sys
sys.path.insert(0, "eval")
from common import load, report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def keyword(text):
    t = text.lower()
    if any(k in t for k in ["documentation", "readme", "docstring", "docs say", "typo"]): return "docs"
    if any(k in t for k in ["how do i", "how to", "how can i", "is it possible", "?"]): return "question"
    if any(k in t for k in ["error", "crash", "exception", "traceback", "fails", "broken", "regression"]): return "bug"
    if any(k in t for k in ["feature", "support for", "would be nice", "proposal", "add "]): return "feature"
    return "bug"

train, test = load("train"), load("test")
y = [r["label"] for r in test]

t0 = time.time(); p = [keyword(r["text"]) for r in test]
report("Keyword rules", y, p, time.time() - t0)

clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit([r["text"] for r in train], [r["label"] for r in train])
t0 = time.time(); p = clf.predict([r["text"] for r in test])
report("TF-IDF + LogReg", y, list(p), time.time() - t0)