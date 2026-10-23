import sys
sys.path.insert(0, "bot")
from retrieval import DocsIndex

idx = DocsIndex(open("data/demo_README.md").read())
print(len(idx.chunks), "chunks:", [c["heading"] for c in idx.chunks])
for q in ["How do I install Flask?",
          "How do I run a minimal app?",
          "Where can I get help or ask questions?",
          "How do I configure database migrations with SQLAlchemy?",   # not in README
          "What is the best pizza topping?"]:                          # not in README
    chunk, score = idx.search(q)
    print(f"\nQ: {q}\n score={score:.3f} -> section: {chunk['heading']}")