import re
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def chunk_markdown(text, max_words=180):
    """Split by markdown headings (ignoring # lines inside code fences), then cut to ~max_words."""
    sections, current, in_code = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,4} ", line) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks = []
    for sec in sections:
        lines = sec.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else "Intro"
        words = sec.split()
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            if len(piece.split()) >= 15:
                chunks.append({"heading": heading, "text": piece})
    return chunks


class DocsIndex:
    def __init__(self, markdown_text):
        self.chunks = chunk_markdown(markdown_text)
        texts = [f"{c['heading']}\n{c['text']}" for c in self.chunks]
        self.emb = _get_model().encode(texts, normalize_embeddings=True, convert_to_tensor=True)

    def search(self, question):
        q = _get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True,
                                convert_to_tensor=True)
        scores = util.cos_sim(q, self.emb)[0]
        i = int(scores.argmax())
        return self.chunks[i], float(scores[i])