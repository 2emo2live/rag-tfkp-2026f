#!/usr/bin/env python3
"""Convert tfkp_15_qa.md into dev_questions.jsonl and a draft dev_gold.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TYPE_MAP = {
    "условия": "conditions",
    "сопоставление": "matching",
    "применение": "calculations",
    "недостаточность данных": "insufficient",
}

# Curated sets of corpus blocks that together justify each answer.
# Several alternatives are allowed because the evaluator takes the best F1.
GOLD_CITATIONS = {
    "q_001": [
        ["b_p011_001"],
        ["b_p011_001", "b_p011_002"],
        ["b_p011_001", "b_p011_002", "b_p011_003"],
    ],
    "q_002": [
        ["b_p012_002"],
        ["b_p012_001", "b_p012_002"],
    ],
    "q_003": [
        ["b_p013_001", "b_p013_002"],
        ["b_p011_001", "b_p013_001", "b_p013_002"],
    ],
    "q_004": [
        ["b_p028_002", "b_p031_001", "b_p031_002"],
        ["b_p028_002", "b_p031_001"],
        ["b_p026_002", "b_p028_002", "b_p031_002"],
    ],
    "q_005": [
        ["b_p030_002"],
        ["b_p030_001", "b_p030_002"],
    ],
    "q_006": [
        ["b_p033_001"],
        ["b_p031_003", "b_p033_001"],
    ],
    "q_007": [
        ["b_p034_001"],
        ["b_p034_001", "b_p020_002"],
    ],
    "q_008": [["b_p035_001"]],
    "q_009": [
        ["b_p013_001", "b_p034_001"],
        ["b_p011_001", "b_p013_001", "b_p034_001"],
    ],
    "q_010": [
        ["b_p036_001"],
        ["b_p036_001", "b_p037_001"],
    ],
    "q_011": [["b_p037_002"]],
    "q_012": [
        ["b_p038_002", "b_p021_001"],
        ["b_p038_002"],
    ],
    "q_013": [],
    "q_014": [],
    "q_015": [],
}


def parse_questions(md_text: str) -> list[dict]:
    sections = re.split(r"^##\s+", md_text, flags=re.MULTILINE)[1:]
    out: list[dict] = []
    for section in sections:
        header, _, body = section.partition("\n")
        header = header.strip()
        num_match = re.match(r"(\d+)\.", header)
        if not num_match:
            continue
        qid = f"q_{int(num_match.group(1)):03d}"

        type_match = re.search(r"\*\*Тип:\*\*\s*([^.]+)\.", body)
        qtype = type_match.group(1).strip() if type_match else ""
        category = TYPE_MAP.get(qtype, qtype)

        q_match = re.search(
            r"\*\*Вопрос\.\*\*\s*(.*?)(?=\n\s*-\s*\*\*[A-Z]\.\*\*)",
            body,
            flags=re.DOTALL,
        )
        question = re.sub(r"\s+", " ", q_match.group(1)).strip() if q_match else ""

        options: list[dict] = []
        for opt_match in re.finditer(
            r"-\s*\*\*([A-Z])\.\*\*\s*(.*?)(?=\n\s*-\s*\*\*[A-Z]\.\*\*|\n\s*\*\*Ответ)",
            body,
            flags=re.DOTALL,
        ):
            option_text = re.sub(r"\s+", " ", opt_match.group(2)).strip()
            options.append({"id": opt_match.group(1), "text": option_text})

        ans_match = re.search(r"\*\*Ответ:\*\*\s*([^\n]+)", body)
        answer_text = re.sub(r"\s+", " ", ans_match.group(1)).strip() if ans_match else ""

        src_match = re.search(r"\*\*Источник:\*\*\s*(.*?)(?=\n\s*\*\*Типичная ошибка)", body, flags=re.DOTALL)
        source_text = re.sub(r"\s+", " ", src_match.group(1)).strip() if src_match else ""
        pages_match = re.search(r"PDF\s+с\.\s*([\d,\s–—-]+)", source_text)
        pages = (
            [int(x) for x in re.findall(r"\d+", pages_match.group(1))]
            if pages_match
            else []
        )

        rationale = ""
        if ans_match and src_match:
            rationale = body[ans_match.end() : src_match.start()].strip()
            rationale = re.sub(r"^[\s:—-]+", "", rationale)
            rationale = re.sub(r"\s+", " ", rationale).strip()

        abstain = answer_text.lower().startswith("отказ")
        answer_ids = [] if abstain else re.findall(r"\b[A-Z]\b", answer_text)
        out.append(
            {
                "id": qid,
                "title": header,
                "category": category,
                "question": question,
                "options": options,
                "abstain": abstain,
                "answer_ids": sorted(set(answer_ids)),
                "rationale": rationale,
                "source_text": source_text,
                "source_pages": sorted(set(pages)),
            }
        )
    return out


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[А-Яа-яЁёA-Za-z]{3,}", text.lower()))


def choose_candidates(item: dict, corpus: list[dict], limit: int = 5) -> list[dict]:
    pages = set(item["source_pages"])
    query = token_set(item["question"] + " " + item["rationale"])
    scored: list[tuple[float, dict]] = []
    for row in corpus:
        if row["page"] not in pages:
            continue
        text_tokens = token_set(row["text"])
        if not text_tokens:
            continue
        overlap = len(query & text_tokens) / max(1, len(query))
        scored.append((overlap, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qa_md", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--questions-out", type=Path, default=Path("dev_questions.jsonl"))
    parser.add_argument("--gold-out", type=Path, default=Path("dev_gold.jsonl"))
    parser.add_argument("--candidates-out", type=Path, default=Path(".run/dev_candidates.jsonl"))
    args = parser.parse_args()

    items = parse_questions(args.qa_md.read_text(encoding="utf-8"))
    corpus = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]

    with args.questions_out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "options": item["options"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates_out.open("w", encoding="utf-8") as f:
        for item in items:
            candidates = choose_candidates(item, corpus)
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "category": item["category"],
                        "source_pages": item["source_pages"],
                        "candidates": [
                            {
                                "block_id": row["block_id"],
                                "page": row["page"],
                                "section": row["section"],
                                "preview": row["text"][:260],
                            }
                            for row in candidates
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    known_ids = {row["block_id"] for row in corpus}
    with args.gold_out.open("w", encoding="utf-8") as f:
        for item in items:
            citations = [] if item["abstain"] else GOLD_CITATIONS.get(item["id"], [])
            for citation_set in citations:
                missing = set(citation_set) - known_ids
                if missing:
                    raise SystemExit(
                        f"{item['id']}: unknown block ids {sorted(missing)}"
                    )
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "category": item["category"],
                        "abstain": item["abstain"],
                        "answer_ids": item["answer_ids"],
                        "rationale": item["rationale"],
                        "citations": citations,
                        "source_pages": item["source_pages"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Parsed {len(items)} questions -> {args.questions_out}, {args.gold_out}")


if __name__ == "__main__":
    main()
