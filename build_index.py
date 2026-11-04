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

TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+|\d+")


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokenize(text: str, morph: pymorphy3.MorphAnalyzer) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if token.isdigit():
            tokens.append(token)
            continue
        # Latin single-letter variables (f, z, u, v, ...) are meaningful.
        if len(token) == 1 and not re.match(r"[A-Za-z]", token):
            continue
        if token in STOPWORDS:
            continue
        if re.match(r"[А-Яа-яЁё]", token):
            token = morph.parse(token)[0].normal_form
        tokens.append(token)
    return tokens


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

    model = SentenceTransformer(model_name, revision=revision, device="cpu")
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
    tokenized = [tokenize(row["text"], morph) for row in rows]

    # Keep the corpus exactly as supplied (block ids are part of the contract).
    corpus_copy = args.output / "chunks.jsonl"
    with corpus_copy.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output / "bm25_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "block_ids": [row["block_id"] for row in rows],
                "tokens": tokenized,
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
        "format_version": 1,
        "corpus_sha256": file_sha256(args.corpus),
        "chunk_count": len(rows),
        "embedding_model": embedding_cfg["model"],
        "embedding_revision": embedding_cfg.get("revision"),
        "embedding_dim": int(vectors.shape[1]),
        "bm25": {
            "k1": float(retrieval_cfg.get("bm25_k1", 1.5)),
            "b": float(retrieval_cfg.get("bm25_b", 0.75)),
            "tokenizer": "pymorphy3+simple-regex",
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
