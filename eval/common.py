import json
from sklearn.metrics import accuracy_score, f1_score

def load(split):
    return [json.loads(l) for l in open(f"data/{split}.jsonl")]

def report(name, y_true, y_pred, seconds_total):
    row = {"method": name,
           "accuracy": round(accuracy_score(y_true, y_pred), 4),
           "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
           "sec_per_issue": round(seconds_total / len(y_true), 4),
           "n": len(y_true)}
    print(row)
    with open("eval/results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row