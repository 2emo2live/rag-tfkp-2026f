#!/usr/bin/env python3
"""Validate answers.jsonl and trace.jsonl against the assignment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--index", type=Path, help="optional index for block_id checks")
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-llm-calls", type=int, default=4)
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    answers = load_jsonl(args.answers)
    traces = load_jsonl(args.trace)
    errors: list[str] = []

    q_by_id = {q["id"]: q for q in questions}
    if len(q_by_id) != len(questions):
        errors.append("duplicate ids in questions")

    answer_ids = [a.get("id") for a in answers]
    if len(answer_ids) != len(set(answer_ids)):
        errors.append("duplicate answer ids")
    if set(answer_ids) != set(q_by_id):
        missing = sorted(set(q_by_id) - set(answer_ids))
        extra = sorted(set(answer_ids) - set(q_by_id))
        if missing:
            errors.append(f"missing answers: {missing}")
        if extra:
            errors.append(f"extra answers: {extra}")

    valid_blocks: set[str] | None = None
    if args.index:
        valid_blocks = {
            row["block_id"]
            for row in load_jsonl(args.index / "chunks.jsonl")
        }

    for answer in answers:
        qid = answer.get("id")
        question = q_by_id.get(qid)
        if question is None:
            continue
        option_ids = {opt["id"] for opt in question.get("options", [])}
        chosen = answer.get("answer_ids")
        citations = answer.get("citations")
        abstain = answer.get("abstain")
        rationale = answer.get("rationale")
        if not isinstance(chosen, list) or not all(isinstance(x, str) for x in chosen):
            errors.append(f"{qid}: answer_ids must be a list of strings")
            chosen = []
        if len(chosen) != len(set(chosen)):
            errors.append(f"{qid}: duplicate answer_ids")
        if not set(chosen) <= option_ids:
            errors.append(f"{qid}: unknown option ids {sorted(set(chosen) - option_ids)}")
        if not isinstance(citations, list) or not all(isinstance(x, str) for x in citations):
            errors.append(f"{qid}: citations must be a list of strings")
            citations = []
        if len(citations) != len(set(citations)):
            errors.append(f"{qid}: duplicate citations")
        if len(citations) > 4:
            errors.append(f"{qid}: more than 4 citations")
        if valid_blocks is not None and not set(citations) <= valid_blocks:
            errors.append(f"{qid}: unknown block ids {sorted(set(citations) - valid_blocks)}")
        if not isinstance(abstain, bool):
            errors.append(f"{qid}: abstain must be boolean")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{qid}: rationale must be non-empty")
        elif len(rationale) > 1200:
            errors.append(f"{qid}: rationale is {len(rationale)} chars (max 1200)")
        if abstain:
            if chosen or citations:
                errors.append(f"{qid}: abstain must have empty answer_ids and citations")
        else:
            if not chosen:
                errors.append(f"{qid}: non-abstain needs answer_ids")
            if not citations:
                errors.append(f"{qid}: non-abstain needs citations")

    trace_by_id = {t.get("id"): t for t in traces}
    if len(trace_by_id) != len(traces):
        errors.append("duplicate trace ids")
    if set(trace_by_id) != set(q_by_id):
        errors.append("trace ids do not match question ids")
    required = {
        "id",
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "elapsed_ms",
        "retrieved_block_ids",
    }
    for trace in traces:
        qid = trace.get("id")
        missing = required - set(trace)
        if missing:
            errors.append(f"{qid}: trace missing fields {sorted(missing)}")
            continue
        if not isinstance(trace["llm_calls"], int) or trace["llm_calls"] < 1:
            errors.append(f"{qid}: llm_calls must be a positive integer")
        if trace["llm_calls"] > args.max_llm_calls:
            errors.append(f"{qid}: too many llm_calls ({trace['llm_calls']})")
        if trace["input_tokens"] > args.max_input_tokens:
            errors.append(f"{qid}: too many input tokens ({trace['input_tokens']})")
        if trace["output_tokens"] > args.max_output_tokens:
            errors.append(f"{qid}: too many output tokens ({trace['output_tokens']})")
        if not isinstance(trace["retrieved_block_ids"], list):
            errors.append(f"{qid}: retrieved_block_ids must be a list")
        elif valid_blocks is not None and not set(trace["retrieved_block_ids"]) <= valid_blocks:
            errors.append(f"{qid}: trace contains unknown block ids")

    if errors:
        print("FORMAT ERRORS:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print(
        f"OK: {len(answers)} answers, {len(traces)} traces, "
        f"all limits satisfied"
    )


if __name__ == "__main__":
    main()
