"""Fetch a small corpus from the free Wikipedia API (no key needed) and keep
section boundaries so we can build a ground-truth retrieval benchmark."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ragbench-portfolio/0.1 (educational project)"}
SKIP = {"see also", "references", "external links", "further reading", "notes", "bibliography", "sources"}
HEADING = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)

DEFAULT_TITLES = [
    "Transformer (deep learning architecture)",
    "Word embedding",
    "Vector database",
    "Retrieval-augmented generation",
    "Large language model",
    "Attention (machine learning)",
    "Reinforcement learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Gradient descent",
    "Backpropagation",
    "Generative adversarial network",
    "Nearest neighbor search",
    "Tf-idf",
    "Okapi BM25",
    "Support vector machine",
    "Random forest",
    "Natural language processing",
    "Computer vision",
    "Autoencoder",
]


@dataclass
class Section:
    heading: str
    start: int
    end: int


@dataclass
class Document:
    id: str
    title: str
    text: str
    sections: list[Section] = field(default_factory=list)


def parse_sections(text: str) -> list[Section]:
    marks = [(m.start(), m.end(), m.group(2)) for m in HEADING.finditer(text)]
    out = []
    for i, (s, body_start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if name.lower() not in SKIP:
            out.append(Section(name, body_start, end))
    return out


def fetch_article(title: str) -> Document:
    import requests

    params = {
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
        "titles": title, "redirects": 1, "format": "json", "formatversion": 2,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    text = page.get("extract", "")
    return Document(id=page["title"], title=page["title"], text=text, sections=parse_sections(text))


def load_corpus(titles: list[str] | None = None, cache_dir: str = "data/cache") -> list[Document]:
    """Download once, then read from the local JSON cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    docs = []
    for t in titles or DEFAULT_TITLES:
        f = cache / (re.sub(r"[^A-Za-z0-9]+", "_", t) + ".json")
        if f.exists():
            raw = json.loads(f.read_text())
            doc = Document(raw["id"], raw["title"], raw["text"], [Section(**s) for s in raw["sections"]])
        else:
            doc = fetch_article(t)
            f.write_text(json.dumps({"id": doc.id, "title": doc.title, "text": doc.text,
                                     "sections": [s.__dict__ for s in doc.sections]}))
        if doc.text:
            docs.append(doc)
    return docs
