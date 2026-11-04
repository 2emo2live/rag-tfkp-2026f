#!/usr/bin/env python3
"""Evaluate answers against dev/own gold with the assignment's formula."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def f1(predicted: list[str], gold: list[str]) -> float:
    pred = set(predicted)
    ref = set(gold)
    if not pred or not ref:
        return 0.0
    common = len(pred & ref)
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_f1(predicted: list[str], gold_sets: list[list[str]]) -> float:
    if not gold_sets:
        return 0.0
    return max(f1(predicted, gold) for gold in gold_sets)


def evaluate(
    questions: list[dict],
    gold_rows: list[dict],
    answers: list[dict],
    traces: list[dict],
    no_context: bool = False,
) -> dict:
    gold = {row["id"]: row for row in gold_rows}
    pred = {row["id"]: row for row in answers}
    trace = {row["id"]: row for row in traces}
    details = []
    category_scores: dict[str, list[float]] = defaultdict(list)
    costs = []
    latencies = []

    for question in questions:
        qid = question["id"]
        ref = gold[qid]
        out = pred.get(
            qid,
            {"answer_ids": [], "abstain": True, "rationale": "", "citations": []},
        )
        tr = trace.get(qid, {})
        predicted_ids = sorted(set(out.get("answer_ids", [])))
        gold_ids = sorted(set(ref.get("answer_ids", [])))
        citations = list(dict.fromkeys(out.get("citations", [])))
        gold_sets = ref.get("citations", [])
        predicted_abstain = bool(out.get("abstain", False))

        if ref.get("abstain"):
            score = 1.0 if predicted_abstain else 0.0
            answer_correct = score
            citation_f1 = 0.0
        else:
            exact = (not predicted_abstain) and set(predicted_ids) == set(gold_ids)
            answer_correct = 1.0 if exact else 0.0
            citation_f1 = best_f1(citations, gold_sets) if not predicted_abstain else 0.0
            score = answer_correct * (
                1.0 if no_context else (0.8 + 0.2 * citation_f1)
            )

        category = ref.get("category", "unknown")
        category_scores[category].append(score)
        details.append(
            {
                "id": qid,
                "category": category,
                "gold_answer_ids": gold_ids,
                "predicted_answer_ids": predicted_ids,
                "gold_abstain": bool(ref.get("abstain")),
                "predicted_abstain": predicted_abstain,
                "answer_correct": answer_correct,
                "citation_f1": round(citation_f1, 4),
                "score": round(score, 4),
                "gold_citation_sets": gold_sets,
                "predicted_citations": citations,
            }
        )

        input_tokens = int(tr.get("input_tokens", 0))
        output_tokens = int(tr.get("output_tokens", 0))
        llm_calls = int(tr.get("llm_calls", 0))
        costs.append(input_tokens + 4 * output_tokens + 256 * llm_calls)
        if "elapsed_ms" in tr:
            latencies.append(int(tr["elapsed_ms"]))

    category_means = {
        category: statistics.fmean(scores)
        for category, scores in sorted(category_scores.items())
    }
    q_score = statistics.fmean(category_means.values()) if category_means else 0.0
    return {
        "n": len(details),
        "answer_accuracy": statistics.fmean(
            detail["answer_correct"] for detail in details
        )
        if details
        else 0.0,
        "Q": q_score,
        "category_means": category_means,
        "mean_input_tokens": statistics.fmean(
            int(t.get("input_tokens", 0)) for t in traces
        )
        if traces
        else 0.0,
        "mean_output_tokens": statistics.fmean(
            int(t.get("output_tokens", 0)) for t in traces
        )
        if traces
        else 0.0,
        "mean_llm_calls": statistics.fmean(
            int(t.get("llm_calls", 0)) for t in traces
        )
        if traces
        else 0.0,
        "mean_cost": statistics.fmean(costs) if costs else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(
        load_jsonl(args.questions),
        load_jsonl(args.gold),
        load_jsonl(args.answers),
        load_jsonl(args.trace),
        no_context=args.no_context,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
