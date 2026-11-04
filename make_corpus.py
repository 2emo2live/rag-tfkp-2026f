#!/usr/bin/env python3
"""Build a citation-friendly corpus.jsonl from the TFKP textbook PDF.

The PDF uses Type3 fonts for Cyrillic text (CP1251 byte values exposed as
Latin-1) and Type1 fonts for mathematics.  PyMuPDF's "dict" extraction keeps
the math Unicode and the inter-word spaces, so we decode only Type3 spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pymupdf


# Page numbers are filtered by their y-position in is_page_furniture().
# Do not treat every digit-only line as a header: formulas often put a
# numerator (e.g. "1") and a denominator (e.g. "2" or "2πi") on separate lines.
HEADER_RE = re.compile(
    r"^(ЛЕКЦИИ ПО ТФКП|В\.\s*В\.\s*ГОРЯЙНОВ.*|Содержание)$",
    re.IGNORECASE,
)
SECTION_RE = re.compile(
    r"^(?:§\s*\d+|\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]|Глава\s+\d+)",
)
HYPHEN_CONT_RE = re.compile(r"^[а-яё]")
TARGET_CHARS = 850
MIN_CHARS = 180


CMEX_MAP = {
    "\x00": "(",
    "\x01": ")",
    "\x02": "(",
    "\x03": ")",
    "\x08": "",
    "\x0c": "",
    "\x10": "√",
    "\x11": "",
    "\x12": "(",
    "\x13": ")",
    "\x14": "(",
    "\x15": ")",
    "\x1a": "{",
    "\x1b": "}",
    "\x1e": "|",
}


def decode_char(char: str, font: str) -> str:
    """Decode one PyMuPDF character, fixing the PDF's Type3 CP1251 encoding."""
    if re.fullmatch(r"F\d+", font):
        char = char.encode("latin-1", "replace").decode("cp1251", "replace")
        char = (
            char.replace("\x16", "—")
            .replace("\x15", "-")
            .replace("\x9f", "§")
            .replace("џ", "§")
        )
        if char == "\x01":
            char = "е"
        if ord(char) < 32 and char not in "\n\t":
            char = " "
    elif font.startswith("CMEX"):
        char = CMEX_MAP.get(char, "")
    return char


def page_bars(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """Short horizontal rules that mark complex conjugation (a bar over z)."""
    bars: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if abs(rect.height) < 0.2 and 4.0 <= rect.width <= 7.0:
            bars.append((float(rect.x0), float(rect.x1), float(rect.y0)))
    return bars


def line_text(line: dict, bars: list[tuple[float, float, float]] | None = None) -> str:
    """Reconstruct a line, restoring overbars drawn as separate PDF paths."""
    bars = bars or []
    out: list[str] = []
    for span in line["spans"]:
        font = span.get("font", "")
        for char in span.get("chars", []):
            text = decode_char(char["c"], font)
            x0, y0, x1, _ = char["bbox"]
            center = (x0 + x1) / 2.0
            for bar_x0, bar_x1, bar_y in bars:
                if (
                    bar_x0 - 0.5 <= center <= bar_x1 + 0.5
                    and y0 - 1.0 <= bar_y <= y0 + 4.0
                ):
                    text += "\u0305"
                    break
            out.append(text)
    return "".join(out).strip()


def line_size(line: dict) -> float:
    return max((float(span.get("size", 0.0)) for span in line["spans"]), default=0.0)


def is_page_furniture(text: str, y: float, page_height: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if y < 45.0 and (
        stripped.isdigit()
        or stripped.startswith("ЛЕКЦИИ")
        or "ГОРЯЙНОВ" in stripped
        or stripped == "Содержание"
    ):
        return True
    if y > page_height - 45.0 and stripped.isdigit():
        return True
    if HEADER_RE.fullmatch(stripped):
        return True
    return False


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = text.strip()
    # Fractions in the PDF are split into a numerator and a denominator line.
    text = re.sub(r"(∂[A-Za-z])\s+(∂[A-Za-z])", r"\1/\2", text)
    text = re.sub(r"\bf\s+′", "f′", text)
    text = re.sub(r"\b([A-Za-z])\s+′", r"\1′", text)
    return text


def split_long_line(text: str, limit: int = TARGET_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    out: list[str] = []
    buf = ""
    for part in parts:
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= limit:
            buf += " " + part
        else:
            out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    # Extremely long formula/paragraph without sentence punctuation.
    final: list[str] = []
    for part in out:
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            final.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final


def append_line(buf: list[str], line: str) -> None:
    if not line:
        return
    if buf and buf[-1].endswith("-") and HYPHEN_CONT_RE.match(line):
        buf[-1] = buf[-1][:-1] + line
    else:
        buf.append(line)


def flush_chunk(
    buf: list[str],
    page: int | None,
    section: str,
    chunks: list[dict],
) -> None:
    if not buf or page is None:
        return
    text = clean_text("\n".join(buf))
    if not text:
        return
    chunks.append({"page": page, "section": section, "text": text})
    buf.clear()


def merge_short_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    merged: list[dict] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk["text"]) < MIN_CHARS
            and len(merged[-1]["text"]) + 1 + len(chunk["text"]) <= TARGET_CHARS * 2
        ):
            merged[-1]["text"] += "\n" + chunk["text"]
            continue
        if (
            merged
            and len(merged[-1]["text"]) < MIN_CHARS
            and len(merged[-1]["text"]) + 1 + len(chunk["text"]) <= TARGET_CHARS * 2
        ):
            chunk["text"] = merged[-1]["text"] + "\n" + chunk["text"]
            merged[-1] = chunk
            continue
        merged.append(chunk)
    return merged


def repair_hyphen_splits(chunks: list[dict]) -> None:
    """Move a word tail split by a chunk boundary into the next chunk.

    Chunks are flushed when they reach TARGET_CHARS, which can happen in the
    middle of a hyphenated word.  The word is complete in the source text but
    the hyphen and its tail land in different chunks.  Moving the tail keeps
    the chunk count and block ids unchanged; only the text distribution
    between two adjacent chunks changes.
    """
    for i in range(len(chunks) - 1):
        left = chunks[i]["text"]
        right = chunks[i + 1]["text"]
        match = re.search(r"([А-Яа-яЁёA-Za-z0-9]+)-$", left.rstrip())
        if not match:
            continue
        right_stripped = right.lstrip()
        if not HYPHEN_CONT_RE.match(right_stripped):
            continue
        new_left = left.rstrip()[: match.start()].rstrip()
        if not new_left:
            continue
        chunks[i]["text"] = new_left
        chunks[i + 1]["text"] = match.group(1) + right_stripped


def extract_chunks(pdf_path: Path, first_page: int, last_page: int) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    chunks: list[dict] = []
    buf: list[str] = []
    buf_page: int | None = None
    section = ""

    for page_index in range(first_page - 1, min(last_page, doc.page_count)):
        page = doc[page_index]
        page_no = page_index + 1
        height = float(page.rect.height)
        bars = page_bars(page)
        blocks = page.get_text("rawdict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = line_text(line, bars)
                if not text:
                    continue
                y = float(line["bbox"][1])
                if is_page_furniture(text, y, height):
                    continue
                size = line_size(line)
                is_heading = bool(SECTION_RE.match(text)) and (
                    size >= 11.0 or len(text) < 90
                )
                if is_heading:
                    flush_chunk(buf, buf_page, section, chunks)
                    section = text
                    for part in split_long_line(text):
                        buf.append(part)
                    buf_page = page_no
                    continue
                if buf_page is None:
                    buf_page = page_no
                for part in split_long_line(text):
                    append_line(buf, part)
                current_len = len(clean_text("\n".join(buf)))
                if current_len >= TARGET_CHARS:
                    flush_chunk(buf, buf_page, section, chunks)
                    buf_page = None
        flush_chunk(buf, buf_page, section, chunks)
        buf_page = None

    chunks = merge_short_chunks(chunks)
    repair_hyphen_splits(chunks)

    # Stable, page-readable IDs.
    per_page: dict[int, int] = {}
    for chunk in chunks:
        page_no = int(chunk["page"])
        per_page[page_no] = per_page.get(page_no, 0) + 1
        chunk["block_id"] = f"b_p{page_no:03d}_{per_page[page_no]:03d}"
    return chunks


def write_outputs(chunks: Iterable[dict], corpus_path: Path, manifest_path: Path) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as corpus_f, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_f:
        for chunk in chunks:
            row = {
                "block_id": chunk["block_id"],
                "page": chunk["page"],
                "section": chunk["section"],
                "text": chunk["text"],
            }
            corpus_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_f.write(
                json.dumps(
                    {
                        "block_id": chunk["block_id"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "char_count": len(chunk["text"]),
                        "text_sha256": hashlib.sha256(
                            chunk["text"].encode("utf-8")
                        ).hexdigest(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the textbook PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus.jsonl"),
        help="Output corpus JSONL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus_manifest.jsonl"),
        help="Output manifest JSONL (no full text)",
    )
    parser.add_argument("--first-page", type=int, default=3)
    parser.add_argument("--last-page", type=int, default=147)
    args = parser.parse_args()

    chunks = extract_chunks(args.pdf, args.first_page, args.last_page)
    write_outputs(chunks, args.output, args.manifest)
    total_chars = sum(len(c["text"]) for c in chunks)
    print(
        f"Wrote {len(chunks)} chunks, {total_chars} chars "
        f"to {args.output}; manifest: {args.manifest}"
    )


if __name__ == "__main__":
    main()
