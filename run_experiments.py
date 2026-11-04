#!/usr/bin/env python3
"""Run the three required configurations and two ablations on the dev set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


MODES = [
    ("no_context", "no_context"),
    ("rag_single", "rag_single"),
    ("rag_bm25_dual", "rag_bm25_dual"),
    ("full", "full"),
    ("ablation_no_hybrid", "no_hybrid"),
    ("ablation_no_reranker", "no_reranker"),
]


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--questions", type=Path, default=Path("dev_questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("dev_gold.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--results", type=Path, default=Path("experiments"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--modes", nargs="*", default=[name for name, _ in MODES])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    args.results.mkdir(parents=True, exist_ok=True)
    rows = []

    selected = dict(MODES)
    for name in args.modes:
        if name not in selected:
            raise SystemExit(f"unknown experiment {name}")
        mode = selected[name]
        out_dir = args.results / name
        out_dir.mkdir(parents=True, exist_ok=True)
        answers = out_dir / "answers.jsonl"
        trace = out_dir / "trace.jsonl"
        predict_cmd = [
            sys.executable,
            str(root / "predict.py"),
            "--index",
            str((root / args.index).resolve()),
            "--questions",
            str((root / args.questions).resolve()),
            "--output",
            str(answers.resolve()),
            "--trace",
            str(trace.resolve()),
            "--config",
            str((root / args.config).resolve()),
            "--mode",
            mode,
        ]
        if args.limit:
            predict_cmd.extend(["--limit", str(args.limit)])
        run(predict_cmd, env)

        eval_json = out_dir / "eval.json"
        eval_cmd = [
            sys.executable,
            str(root / "evaluate.py"),
            "--questions",
            str((root / args.questions).resolve()),
            "--gold",
            str((root / args.gold).resolve()),
            "--answers",
            str(answers.resolve()),
            "--trace",
            str(trace.resolve()),
            "--output",
            str(eval_json.resolve()),
        ]
        if mode == "no_context":
            eval_cmd.append("--no-context")
        run(eval_cmd, env)
        metrics = json.loads(eval_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "answer_accuracy": f"{metrics['answer_accuracy']:.4f}",
                "Q": "" if mode == "no_context" else f"{metrics['Q']:.4f}",
                "mean_input_tokens": f"{metrics['mean_input_tokens']:.2f}",
                "mean_output_tokens": f"{metrics['mean_output_tokens']:.2f}",
                "mean_llm_calls": f"{metrics['mean_llm_calls']:.2f}",
                "mean_cost": f"{metrics['mean_cost']:.2f}",
                "median_latency_ms": f"{metrics['median_latency_ms']:.0f}",
            }
        )

    csv_path = root / "experiments.csv"
    fieldnames = [
        "name",
        "answer_accuracy",
        "Q",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_llm_calls",
        "mean_cost",
        "median_latency_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
