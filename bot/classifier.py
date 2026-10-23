import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MB_MODEL_DIR", "models/maintainerbot-distilbert")


class Classifier:
    def __init__(self, path=MODEL_DIR):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(self, text):
        enc = self.tok(text[:1500], truncation=True, max_length=256,
                       return_tensors="pt", return_token_type_ids=False)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, -1)[0]
        i = int(probs.argmax())
        return self.model.config.id2label[i], float(probs[i])