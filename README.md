# MaintainerBot

Maintainers answer the same questions all day. This bot sorts issues, points to the docs that already answer them, and never does anything you can't undo.

## What it does
- Labels new issues (bug, feature, question, docs) with a fine-tuned DistilBERT classifier
- Finds the README section that answers a question (bge-small embeddings) and replies with Gemma, or says "not documented"
- Only adds labels (`mb:*`) and comments. It never merges or closes anything.
- Dry-run by default, every decision logged to an audit file, one command to undo

## Results (held-out repo: numpy)
| Method | Accuracy | Macro-F1 |
|---|---|---|
| Keyword rules | 0.512 | 0.541 |
| TF-IDF + LogReg | 0.507 | 0.572 |
| Zero-shot gemma3:1b | 0.353 | 0.386 |
| Fine-tuned DistilBERT | 0.614 | 0.668 |

Model: https://huggingface.co/shine-sri/maintainerbot-distilbert

## Built with
- Claude (Anthropic): coding assistant used during development
- Gemma (open-weight, via Ollama): reply text
- DistilBERT, fine-tuned: issue classifier
- bge-small (sentence-transformers): docs search
- Hugging Face, scikit-learn, GitHub Actions, Google Colab

## License
MIT