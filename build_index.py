#!/usr/bin/env python3
"""Build the retrieval index (BM25 tokens + dense embeddings).

The index is intentionally small and self-contained: a JSONL copy of the
corpus, lemmatised BM25 tokens and a float32 embedding matrix.  Everything is
written under --output, so the command can be run from any working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pymorphy3

STOPWORDS = {
    "а",
    "без",
    "более",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "вы",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "здесь",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "когда",
    "кто",
    "ли",
    "либо",
    "мне",
    "может",
    "мы",
    "на",
    "над",
    "надо",
    "наш",
    "не",
    "него",
    "нее",
    "нет",
    "ни",
    "них",
    "но",
    "ну",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "очень",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "также",
    "такой",
    "там",
    "те",
    "тем",
    "то",
    "того",
    "тоже",
    "той",
    "только",
    "том",
    "ты",
    "у",
    "уже",
    "хотя",
    "чего",
    "чей",
    "чем",
    "что",
    "чтобы",
    "чье",
    "эта",
    "эти",
    "это",
    "этого",
    "этой",
    "этом",
    "этот",
    "я",
}

# Prose and formulae are deliberately indexed separately.  Indexing '=' or
# '∂' as standalone terms makes almost every mathematical chunk look alike.
TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+")
FORMULA_ATOM_RE = re.compile(
    r"partial|sqrt|pi|[a-z]+bar|[a-z]+prime\d*|[a-z]+|\d+|[=+\-*/^()]"
)
FORMULA_MARKER_RE = re.compile(
    r"partial|sqrt|pi|[a-z]+bar|[a-z]+prime\d*|[∫∑∏∂√∞=+\-−*/^′″̅]"
)
FORMULA_OPERATORS = {"=", "+", "-", "*", "/", "^", "(", ")"}


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokenize_words(text: str, morph: pymorphy3.MorphAnalyzer) -> list[str]:
    """Lemmatise natural-language text for the prose BM25 index."""
    tokens: list[str] = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if token.isdigit():
            tokens.append(token)
            continue
        # Latin variables in prose are useful; isolated Cyrillic letters are not.
        if len(token) == 1 and not re.match(r"[A-Za-z]", token):
            continue
        if re.match(r"[А-Яа-яЁё]", token):
            token = morph.parse(token)[0].normal_form
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


# Preserve the previous public function name for callers importing tokenize.
tokenize = tokenize_words


def normalize_formula_text(text: str) -> str:
    """Canonicalise common PDF/LaTex spellings before extracting features."""
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace(r"\partial", " partial ")
    text = text.replace(r"\pi", " pi ")
    text = text.replace(r"\sqrt", " sqrt ")
    text = re.sub(r"\\bar\s*\{\s*([a-z])\s*\}", r" \1bar ", text)
    text = re.sub(r"\\overline\s*\{\s*([a-z])\s*\}", r" \1bar ", text)
    text = re.sub(r"\\bar\s+([a-z])", r" \1bar ", text)
    text = re.sub(r"\\overline\s+([a-z])", r" \1bar ", text)
    # The PDF extractor represents a bar as the combining U+0305 character.
    text = re.sub(r"([a-z])\u0305", r" \1bar ", text)
    text = re.sub(r"([a-z])[′']{2}", r" \1prime2 ", text)
    text = re.sub(r"([a-z])[′']", r" \1prime ", text)
    text = text.replace("∂", " partial ").replace("√", " sqrt ")
    text = text.replace("π", " pi ").replace("−", "-")
    return re.sub(r"[ \t]+", " ", text)


def formula_ngrams(text: str) -> list[str]:
    """Generate formula-aware 2--3-gram BM25 features.

    Formula operators identify an expression but never become independent
    tokens.  For instance, both ∂f/∂z̅ and its LaTex spelling emit
    ``math:partial_f_partial`` and ``math:f_partial_zbar``.
    """
    features: list[str] = []
    for line in normalize_formula_text(text).splitlines():
        if not FORMULA_MARKER_RE.search(line):
            continue
        atoms = [
            atom for atom in FORMULA_ATOM_RE.findall(line)
            if atom not in FORMULA_OPERATORS
        ]
        if not atoms:
            continue
        for atom in atoms:
            if atom in {"partial", "sqrt", "pi"} or atom.endswith(("bar", "prime", "prime2")):
                features.append(f"math:{atom}")
        for n in (2, 3):
            for start in range(len(atoms) - n + 1):
                features.append("math:" + "_".join(atoms[start : start + n]))
    return features


def load_corpus(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            for key in ("block_id", "page", "section", "text"):
                if key not in row:
                    raise SystemExit(f"{path}:{line_no}: missing field {key!r}")
            rows.append(
                {
                    "block_id": str(row["block_id"]),
                    "page": int(row["page"]),
                    "section": str(row.get("section", "")),
                    "text": str(row["text"]),
                }
            )
    if not rows:
        raise SystemExit(f"{path}: empty corpus")
    ids = [row["block_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate block_id values")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_embeddings(
    texts: Iterable[str],
    model_name: str,
    revision: str | None,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision, device="cpu", local_files_only=True)
    passages = ["passage: " + text for text in texts]
    vectors = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    embedding_cfg = config["embedding"]
    retrieval_cfg = config["retrieval"]

    rows = load_corpus(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)

    morph = pymorphy3.MorphAnalyzer()
    text_tokens = [tokenize_words(row["text"], morph) for row in rows]
    math_tokens = [formula_ngrams(row["text"]) for row in rows]

    # Keep the corpus exactly as supplied (block ids are part of the contract).
    corpus_copy = args.output / "chunks.jsonl"
    with corpus_copy.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output / "bm25_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "block_ids": [row["block_id"] for row in rows],
                # ``tokens`` is retained for backwards compatibility with
                # earlier indexes; new readers use the explicit field names.
                "tokens": text_tokens,
                "text_tokens": text_tokens,
                "formula_tokens": math_tokens,
                "k1": float(retrieval_cfg.get("bm25_k1", 1.5)),
                "b": float(retrieval_cfg.get("bm25_b", 0.75)),
            },
            f,
            ensure_ascii=False,
        )

    vectors = build_embeddings(
        [row["text"] for row in rows],
        embedding_cfg["model"],
        embedding_cfg.get("revision"),
        int(embedding_cfg.get("batch_size", 32)),
    )
    np.save(args.output / "embeddings.npy", vectors)

    meta = {
        "format_version": 2,
        "corpus_sha256": file_sha256(args.corpus),
        "chunk_count": len(rows),
        "embedding_model": embedding_cfg["model"],
        "embedding_revision": embedding_cfg.get("revision"),
        "embedding_dim": int(vectors.shape[1]),
        "bm25": {
            "k1": float(retrieval_cfg.get("bm25_k1", 1.5)),
            "b": float(retrieval_cfg.get("bm25_b", 0.75)),
            "tokenizer": "pymorphy3 prose + formula 2-3grams",
        },
    }
    (args.output / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Indexed {len(rows)} blocks, {vectors.shape[1]}-d embeddings "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
