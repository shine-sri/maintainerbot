# MaintainerBot

> Maintainers answer the same questions all day. This bot sorts issues, points to the docs that already answer them, and never does anything you can't undo.

MaintainerBot is an open-source GitHub Action. When an issue is opened, a fine-tuned classifier labels it, embedding search looks for a README section that answers it, and Gemma writes a short friendly reply from that section only. If the bot is unsure, it does nothing.

<!-- TODO: add demo GIF: ![demo](docs/demo.gif) -->

## What it does

| Task | Who decides | What Gemma writes |
|---|---|---|
| Issue triage | Fine-tuned DistilBERT labels the issue: bug, feature, question or docs | Nothing |
| Docs answer | Embedding search (bge-small) finds the best README section; a threshold decides whether to answer | A short answer using only that section, or "not documented" |

**Small trained models make every decision. Gemma only writes the text of the reply.** That keeps the results measurable and stops a strange Gemma answer from causing a wrong label.

Example: someone opens "How do I install this on Windows?". The bot adds the label `mb:question`, finds the installation section of the README and comments with the answer. If the README says nothing about it, the bot says "not documented" and leaves it to a human.

## Safety design

| Feature | How it works |
|---|---|
| Dry-run by default | The bot only logs what it would do until `MAINTAINERBOT_LIVE=true` is set |
| Minimal permissions | The workflow has `permissions: issues: write` and nothing else |
| Reversible actions only | Labels (`mb:*`) and comments. It never merges, closes or deletes |
| Audit log | One JSONL line per decision: time, issue, label, confidence, retrieval score, action, reason. Do-nothing cases are logged too |
| Undo | `python bot/undo.py --yes` removes every `mb:` label and every comment carrying the bot's marker |
| Abstain when unsure | Below the classifier threshold (0.80) the bot does nothing. Below the retrieval threshold (0.70) it says "not documented" |
| Untrusted input | Issue text is treated as data. The classifier can only return one of four labels, and Gemma's output is only ever a comment, never a command |
| Secrets | Keys are never in the repo. The workflow uses the built-in `GITHUB_TOKEN` |

## Results

Test set: **682 closed issues from numpy**, a repository the model never saw during training (repository-held-out split).

| Method | Accuracy | Macro-F1 | Time per issue |
|---|---|---|---|
| Keyword rules | 0.512 | 0.541 | ~0 s |
| TF-IDF + logistic regression | 0.507 | 0.572 | ~0 s |
| Zero-shot Gemma 3 1B (300-issue sample) | 0.353 | 0.386 | 0.37 s |
| **Fine-tuned DistilBERT (ours)** | **0.614** | **0.668** | 0.044 s (CPU) |

Macro-F1 is the headline metric. About 53% of the test set is questions, so plain accuracy can mislead.

### Confidence threshold trade-off

Raising the threshold makes the bot answer fewer issues but get more of them right.

| Threshold | Coverage | Accuracy on answered | Macro-F1 on answered |
|---|---|---|---|
| 0.0 | 100% | 0.614 | 0.668 |
| 0.7 | 88% | 0.649 | 0.701 |
| **0.8 (default)** | **81%** | **0.653** | **0.702** |
| 0.9 | 68% | 0.717 | 0.752 |
| 0.95 | 55% | 0.766 | 0.778 |

### Confusion matrix (rows = true, columns = predicted)

| | bug | feature | question | docs |
|---|---|---|---|---|
| **bug** | 126 | 3 | 10 | 2 |
| **feature** | 5 | 66 | 2 | 2 |
| **question** | 178 | 26 | 129 | 29 |
| **docs** | 4 | 1 | 1 | 98 |

### Error analysis

- **Questions predicted as bugs (178 of 362).** Many numpy question issues contain tracebacks and code, and the model also learned from repos where question issues look like that. Bug precision is only 0.40 and question recall is 0.36.
- **The reverse also happens.** A clear bug report with a traceback written for a Flask demo was labeled `question` at 96% confidence. A confidence threshold cannot catch a confidently wrong label.
- **Why it is not harmful.** A wrong label is cheap and reversible, and the docs check refused to reply (retrieval score 0.657 is below 0.70), so no wrong answer was posted.
- **Short or vague issues** (for example "help") get low confidence and the bot abstains.

<!-- TODO: add 1-2 more cases from eval/test_preds.jsonl -->

## How it works

```
Issue opened
   -> DistilBERT picks a label + confidence
   -> confidence >= 0.80 ?  no -> do nothing
   -> label == question ?   yes -> bge-small finds the best README section
   -> score >= 0.70 ?       no -> "not documented", left to a human
   -> Gemma writes a reply from that section only (template fallback if it fails)
   -> dry-run: log only | live: add label + comment
   -> audit log records the decision and its reason
```

## Install it in your repo

1. Copy `.github/workflows/maintainerbot.yml` and the `bot/` folder into your repository. Your repo needs a `README.md` with the docs you want the bot to answer from.
2. In `maintainerbot.yml`, make sure the model download uses `shine-sri/maintainerbot-distilbert`.
3. Push. The bot starts in **dry-run** mode and only writes the audit log (download it from the run's artifacts).
4. When you are happy with its decisions, go to Settings, then Secrets and variables, then Actions, then Variables, and add `MAINTAINERBOT_LIVE` with the value `true`.
5. To remove everything the bot added:
   ```bash
   export GITHUB_TOKEN=<token with Issues read/write on the repo>
   export GITHUB_REPOSITORY=owner/repo
   python bot/undo.py          # lists what would be removed
   python bot/undo.py --yes    # removes it
   ```

Settings (environment variables in the workflow):

| Variable | Default | Meaning |
|---|---|---|
| `MB_LIVE` | `false` | Post labels and comments for real |
| `MB_CLS_THRESHOLD` | `0.8` | Minimum classifier confidence |
| `MB_DOC_THRESHOLD` | `0.70` | Minimum docs match score |
| `MB_DOCS_PATH` | `README.md` | Docs the bot may answer from |
| `MB_GEMMA` | `ollama` | `off` always uses the templated reply (used in Actions, which has no Ollama) |

## Run it locally

```bash
pip install -r requirements.txt
ollama pull gemma3:1b            # optional: Gemma replies
streamlit run app.py             # interactive demo, never posts anything
```

Reproduce the results:

```bash
python data/download_issues.py kyverno/kyverno pallets/flask psf/requests
python data/download_labeled.py
python data/build_dataset.py --test numpy__numpy --val-frac 0.15
python eval/baselines.py
python eval/zero_shot_gemma.py --model gemma3:1b --n 300
python eval/eval_finetuned.py
```

Training notebook: `train/finetune_distilbert.ipynb` (Google Colab, T4 GPU, about 1 minute).

## Data and training

- Closed issues from kyverno, flask, requests, pandas, scikit-learn and numpy, with labels mapped to bug, feature, question and docs. Issues with no usable label, or with labels from more than one class, were dropped.
- Train: 1,883 issues. Validation: 330 (a stratified slice of the training repos). Test: 682 (numpy only).
- DistilBERT (`distilbert-base-uncased`), 3 epochs, class-weighted loss because docs is the smallest class (114 training examples).

## Limitations

- Labels are noisy, because different maintainers label differently.
- Validation scores (macro-F1 about 0.94) are much higher than held-out test scores (0.668). Validation shares repositories with training, so the test score is the honest one.
- The model may not work as well on repositories it has not seen. Question issues containing code or tracebacks are often labeled as bugs.
- Only the first 256 tokens of an issue are read.
- It does not write fluent answers to questions the docs do not cover, and it says so.
- Gemma (1B) sometimes adds a sentence that is not in the docs. It only writes text and never makes decisions.
- Zero-shot Gemma was scored on a 300-issue sample, and unparseable outputs count as `bug`.
- Evaluated on one held-out repository. Results on other projects will differ.
- English only.
- <!-- TODO: shadow replay not run yet; add "agreed with maintainers on X% of N historical issues" -->

## Built with

AI tools used in this project (hackathon rule):

- **Claude (Anthropic)**: coding assistant used throughout development
- **Gemma** (open-weight, run locally with Ollama): reply text
- **DistilBERT** (open-weight, fine-tuned by me): issue classifier
- **bge-small-en-v1.5** (open-weight, sentence-transformers): docs search

Also: Hugging Face Transformers, scikit-learn, Streamlit, GitHub Actions, Google Colab.

## Model

https://huggingface.co/shine-sri/maintainerbot-distilbert

## License

MIT. See [LICENSE](LICENSE).
