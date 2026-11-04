#!/usr/bin/env python3
"""Answer TFKP multiple-choice questions with a local evidence-based RAG.

Modes (the first three are the required comparison, the last two are ablations):
  final           hybrid BM25+dense, reranker, neighbour context
  rag_single      one BM25 search, no reranker, no neighbours
  rag_bm25_dual   text BM25 + formula BM25 quota, no dense/reranker/neighbours
  no_context      the LLM answers without any retrieved fragments
  no_hybrid       BM25 only + reranker + neighbours
  no_reranker     hybrid retrieval without the cross-encoder reranker
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from build_index import formula_ngrams, tokenize
import pymorphy3


MODES = {
    "final",
    "rag_single",
    "rag_bm25_dual",
    "no_context",
    "no_hybrid",
    "no_reranker",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def approx_tokens(text: str) -> int:
    """Conservative estimate for Russian text without loading a tokenizer."""
    return max(1, math.ceil(len(text) / 2.5))


def format_options(options: list[dict]) -> str:
    return "\n".join(f"{opt['id']}. {opt['text']}" for opt in options)


def build_context(rows: list[dict], indices: list[int], max_chars: int) -> tuple[str, list[str]]:
    parts: list[str] = []
    used: list[str] = []
    total = 0
    for idx in indices:
        row = rows[idx]
        header = (
            f"[ФРАГМЕНТ {row['block_id']} | PDF с. {row['page']}"
            + (f" | {row['section']}" if row.get("section") else "")
            + "]"
        )
        text = row["text"].strip()
        block = header + "\n" + text
        if used and total + len(block) + 2 > max_chars:
            continue
        if len(block) > max_chars:
            block = block[:max_chars]
        parts.append(block)
        used.append(row["block_id"])
        total += len(block) + 2
        if total >= max_chars:
            break
    return "\n\n".join(parts), used


def prompt_no_context(question: dict) -> list[dict]:
    system = (
        "Ты — экзаменатор по теории функций комплексного переменного. "
        "Отвечай на вопрос с выбором вариантов, используя свои знания. "
        "Верни только один JSON-объект без markdown."
    )
    user = f"""Вопрос:
{question['question']}

Варианты:
{format_options(question['options'])}

Правила:
1. Выбери ВСЕ верные варианты и только их.
2. Если для однозначного ответа не хватает данных, верни abstain=true и пустые списки.
3. rationale — краткое обоснование по-русски, не более 700 символов.
4. citations оставь пустым списком: в этом режиме фрагменты не предоставляются.

Формат ответа:
{{"answer_ids":["A","C"],"abstain":false,"rationale":"...","citations":[]}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_with_context(question: dict, context: str) -> list[dict]:
    system = (
        "Ты — внимательный экзаменатор по теории функций комплексного переменного. "
        "Ты отвечаешь на вопросы с выбором вариантов, опираясь ИСКЛЮЧИТЕЛЬНО на "
        "фрагменты учебника из сообщения пользователя. Внешние знания и догадки "
        "запрещены. Если фрагментов недостаточно для однозначного ответа, нужно "
        "отказаться. Верни только один JSON-объект без markdown."
    )
    user = f"""Вопрос:
{question['question']}

Варианты:
{format_options(question['options'])}

Фрагменты учебника:
{context}

Правила:
1. Выбери ВСЕ верные варианты и только их; не добавляй варианты без опоры на текст.
2. citations — от 1 до 4 уникальных block_id из фрагментов, которые вместе подтверждают ответ. Лишние ссылки снижают оценку.
3. Если вопрос требует конкретного значения или определённого утверждения, а фрагменты дают только общую теорему и не позволяют выбрать единственный вариант, верни abstain=true, answer_ids=[], citations=[] и объясни в rationale, каких данных не хватает.
4. Если данных достаточно, верни abstain=false, непустые answer_ids и citations.
5. rationale — краткое обоснование по-русски, не более 700 символов. Укажи, почему выбранные варианты верны, а ключевые альтернативы нет.
6. Учитывай, что сопряжение в учебнике может быть записано как z̅ (z с надчеркиванием), а также что формулы могли разбиться при извлечении текста.
7. Выбирай вариант, только если верно именно его собственное утверждение. Не выбирай вариант лишь потому, что он был бы следствием более сильного условия из учебника: например, «только локально, но не глобально» ложно при доказанном глобальном результате, а условие для одного фиксированного семейства не становится достаточным из того, что теорема требует его для всех объектов.
8. Перед возвратом JSON проверь согласованность: каждый выбранный ID должен быть назван верным в rationale, каждый названный в rationale верный ID должен быть в answer_ids, а логически несовместимые варианты нельзя выбрать одновременно.

Ответ строго в формате:
{{"answer_ids":["A","C"],"abstain":false,"rationale":"...","citations":["b_..."]}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(content: str) -> dict | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_answer_ids(value: Any, valid_ids: set[str]) -> list[str]:
    if isinstance(value, str):
        raw = re.findall(r"\b[A-Z]\b", value.upper())
    elif isinstance(value, list):
        raw = []
        for item in value:
            if isinstance(item, str):
                raw.extend(re.findall(r"\b[A-Z]\b", item.upper()))
    else:
        raw = []
    result: list[str] = []
    for item in raw:
        if item in valid_ids and item not in result:
            result.append(item)
    return sorted(result)


def normalize_citations(value: Any, valid_ids: set[str], limit: int) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = [item for item in value if isinstance(item, str)]
    else:
        raw = []
    result: list[str] = []
    for item in raw:
        item = item.strip()
        if item in valid_ids and item not in result:
            result.append(item)
    return result[:limit]


def normalize_rationale(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit].rstrip()


def extract_answer(
    content: str,
    question: dict,
    valid_citations: set[str],
    cfg: dict,
) -> dict:
    data = parse_json_object(content)
    if data is None:
        return {}
    valid_options = {opt["id"] for opt in question["options"]}
    answer_ids = normalize_answer_ids(data.get("answer_ids"), valid_options)
    citations = normalize_citations(
        data.get("citations"), valid_citations, cfg["prompt"]["max_citations"]
    )
    abstain = bool(data.get("abstain", False))
    rationale = normalize_rationale(
        data.get("rationale"), cfg["prompt"]["max_rationale_chars"]
    )
    if abstain:
        answer_ids, citations = [], []
    return {
        "answer_ids": answer_ids,
        "abstain": abstain,
        "rationale": rationale,
        "citations": citations,
    }


def call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    cfg: dict,
    question_id: str,
    submission_id: str,
) -> tuple[str, int, int]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(cfg["llm"]["temperature"]),
        top_p=float(cfg["llm"]["top_p"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
        seed=int(cfg["llm"]["seed"]),
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": bool(cfg["llm"]["enable_thinking"])
            }
        },
        extra_headers={
            "X-Question-ID": question_id,
            "X-Submission-ID": submission_id,
        },
        timeout=float(cfg["llm"]["timeout_s"]),
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return content, input_tokens, output_tokens


def repair_messages(messages: list[dict], bad_content: str) -> list[dict]:
    return messages + [
        {
            "role": "user",
            "content": (
                "Предыдущий ответ не удалось разобрать как JSON. "
                "Верни только корректный JSON-объект указанной схемы, без пояснений "
                "и markdown. Фрагмент предыдущего ответа:\n"
                + bad_content[:1500]
            ),
        }
    ]


class Retriever:
    def __init__(self, index_dir: Path, cfg: dict, mode: str):
        self.rows = load_jsonl(index_dir / "chunks.jsonl")
        self.by_id = {row["block_id"]: i for i, row in enumerate(self.rows)}
        bm25_data = json.loads((index_dir / "bm25_tokens.json").read_text(encoding="utf-8"))
        self.block_ids = bm25_data["block_ids"]
        if self.block_ids != [row["block_id"] for row in self.rows]:
            raise SystemExit("index bm25 token order does not match chunks.jsonl")
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(
            bm25_data.get("text_tokens", bm25_data["tokens"]),
            k1=float(bm25_data.get("k1", 1.5)),
            b=float(bm25_data.get("b", 0.75)),
        )
        formula_tokens = bm25_data.get("formula_tokens")
        if formula_tokens is None:
            raise SystemExit(
                "index has no formula BM25 tokens; rebuild it with build_index.py"
            )
        self.formula_bm25 = BM25Okapi(
            formula_tokens,
            k1=float(bm25_data.get("k1", 1.5)),
            b=float(bm25_data.get("b", 0.75)),
        )
        self.embeddings = np.load(index_dir / "embeddings.npy")
        self.morph = pymorphy3.MorphAnalyzer()
        self.cfg = cfg
        self.mode = mode
        self.embedding_model = None
        self.reranker = None
        if mode in {"final", "no_reranker"}:
            from sentence_transformers import SentenceTransformer

            emb_cfg = cfg["embedding"]
            self.embedding_model = SentenceTransformer(
                emb_cfg["model"], revision=emb_cfg.get("revision"), device="cpu",
                local_files_only=True
            )
        if mode in {"final", "no_hybrid"}:
            from sentence_transformers import CrossEncoder

            rer_cfg = cfg["reranker"]
            self.reranker = CrossEncoder(
                rer_cfg["model"],
                revision=rer_cfg.get("revision"),
                device="cpu",
                max_length=int(rer_cfg.get("max_length", 512)),
                local_files_only=True
            )

    def _bm25_scores(self, question: str) -> np.ndarray:
        tokens = tokenize(question, self.morph)
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)

    def _formula_scores(self, question: str) -> np.ndarray | None:
        tokens = formula_ngrams(question)
        # A formula consisting only of generic variables (f, z, u, ...) is
        # not selective enough to re-rank the prose result.  Use the formula
        # index only when the query has a structural marker that survives PDF
        # extraction, such as a Wirtinger derivative, conjugation or prime.
        discriminative = any(
            token in {"math:partial", "math:sqrt", "math:pi"}
            or "bar" in token
            or "prime" in token
            for token in tokens
        )
        if not tokens or not discriminative:
            return None
        return np.asarray(self.formula_bm25.get_scores(tokens), dtype=np.float64)

    def _dense_scores(self, question: str) -> np.ndarray:
        prefix = self.cfg["embedding"].get("query_prefix", "query: ")
        vector = self.embedding_model.encode(
            [prefix + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return self.embeddings @ vector.astype(np.float32)

    def _top_indices(
        self, scores: np.ndarray, k: int, positive_only: bool = False
    ) -> list[int]:
        if k <= 0:
            return []
        if positive_only:
            candidates = np.flatnonzero(scores > 0)
            if not len(candidates):
                return []
            order = candidates[np.argsort(-scores[candidates], kind="stable")]
        else:
            order = np.argsort(-scores, kind="stable")
        order = order[: min(k, len(order))]
        return [int(i) for i in order]

    def _rrf(
        self, rankings: list[list[int]], weights: list[float] | None = None
    ) -> list[int]:
        rrf_k = float(self.cfg["retrieval"].get("rrf_k", 60))
        if weights is None:
            weights = [1.0] * len(rankings)
        if len(rankings) != len(weights):
            raise ValueError("rankings and weights must have equal lengths")
        fused: dict[int, float] = {}
        for ranking, weight in zip(rankings, weights):
            for rank, idx in enumerate(ranking, 1):
                fused[idx] = fused.get(idx, 0.0) + weight / (rrf_k + rank)
        return sorted(fused, key=lambda idx: (-fused[idx], idx))

    def retrieve(self, question: dict | str) -> tuple[list[int], list[str]]:
        """Retrieve evidence, including formulae that occur only in options.

        The prose/dense query remains the question itself to avoid letting
        distractor wording dominate topical retrieval.  The formula BM25 and
        reranker additionally see the options, because they often contain the
        only distinction between \u2202f/\u2202z and \u2202f/\u2202z\u0305, signs, or exponents.
        """
        if isinstance(question, dict):
            question_text = str(question["question"])
            options_text = format_options(question.get("options", []))
            formula_query = question_text + "\n" + options_text
            rerank_query = question_text + "\n\nВарианты:\n" + options_text
        else:
            question_text = question
            formula_query = question
            rerank_query = question

        ret = self.cfg["retrieval"]
        candidate_k = int(ret.get("candidate_k", 40))
        bm25_ranking = self._top_indices(self._bm25_scores(question_text), candidate_k)
        sparse_rankings = [bm25_ranking]
        sparse_weights = [1.0]
        formula_ranking: list[int] = []
        formula_scores = self._formula_scores(formula_query)
        if formula_scores is not None and self.mode != "rag_single":
            formula_ranking = self._top_indices(
                formula_scores, candidate_k, positive_only=True
            )
            if formula_ranking:
                sparse_rankings.append(formula_ranking)
                sparse_weights.append(float(ret.get("formula_rrf_weight", 0.5)))

        if self.mode == "rag_single":
            fused = bm25_ranking
        elif self.mode in {"rag_bm25_dual", "no_hybrid"}:
            fused = self._rrf(sparse_rankings, sparse_weights)
        else:
            dense_ranking = self._top_indices(self._dense_scores(question_text), candidate_k)
            fused = self._rrf(sparse_rankings + [dense_ranking], sparse_weights + [1.0])

        if self.reranker is not None:
            shortlist = fused[: int(ret.get("rerank_candidates", 24))]
            pairs = [(rerank_query, self.rows[idx]["text"]) for idx in shortlist]
            scores = self.reranker.predict(
                pairs,
                batch_size=int(self.cfg["reranker"].get("batch_size", 16)),
                show_progress_bar=False,
            )
            ranked = [
                idx
                for _, idx in sorted(
                    zip(np.asarray(scores).tolist(), shortlist),
                    key=lambda item: (-item[0], item[1]),
                )
            ]
        else:
            ranked = fused

        top_k = int(ret.get("top_k", 6))
        if self.mode == "rag_bm25_dual" and formula_ranking:
            # RRF alone can displace a very strong prose hit with an unrelated
            # formula hit.  Reserve a small, explicit formula quota instead:
            # text retrieval supplies the topic and formula retrieval can add
            # the exact symbolic condition without replacing that topic.
            formula_slots = min(
                int(ret.get("formula_context_slots", 1)), max(0, top_k - 1)
            )
            selected = list(bm25_ranking[: top_k - formula_slots])
            for idx in formula_ranking:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= top_k:
                        break
            for idx in bm25_ranking:
                if len(selected) >= top_k:
                    break
                if idx not in selected:
                    selected.append(idx)
        else:
            selected = ranked[:top_k]

        if (
            self.mode in {"final", "no_hybrid", "no_reranker"}
            and ret.get("neighbor_expansion", True)
        ):
            selected = self._expand_neighbors(selected, int(ret.get("neighbor_top_n", 4)))

        #selected = sorted(dict.fromkeys(selected))
        selected = list(dict.fromkeys(selected))
        
        context, context_ids = build_context(
            self.rows, selected, int(ret.get("max_context_chars", 9000))
        )
        return selected, context_ids


def _expand_neighbors(self, selected: list[int], top_n: int) -> list[int]:
    result = list(selected)
    seen = set(result)
    for idx in selected[:top_n]:
        row = self.rows[idx]
        for neighbor in (idx - 1, idx + 1):
            if neighbor < 0 or neighbor >= len(self.rows) or neighbor in seen:
                continue
            other = self.rows[neighbor]
            if abs(int(other["page"]) - int(row["page"])) > 1:
                continue
            if other.get("section") and row.get("section") and other["section"] != row["section"]:
                continue
            seen.add(neighbor)
            result.append(neighbor)
    return result


Retriever._expand_neighbors = _expand_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--mode", choices=sorted(MODES), default="final")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    questions = load_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("LLM_MODEL", cfg["llm"]["local_model"])
    api_key = os.environ.get("LLM_API_KEY", "local")
    submission_id = os.environ.get("SUBMISSION_ID", "local-dev")
    client = OpenAI(base_url=base_url, api_key=api_key)

    retriever = None
    if args.mode != "no_context":
        retriever = Retriever(args.index, cfg, args.mode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out_f, args.trace.open(
        "w", encoding="utf-8"
    ) as trace_f:
        for question in questions:
            started = time.perf_counter()
            qid = question["id"]
            context_ids: list[str] = []
            context = ""
            if retriever is not None:
                selected = retriever.retrieve(question)[0]
                context, context_ids = build_context(
                    retriever.rows,
                    selected,
                    int(cfg["retrieval"].get("max_context_chars", 9000)),
                )
                messages = prompt_with_context(question, context)
            else:
                messages = prompt_no_context(question)

            # Keep enough room for the configured completion inside 8192 tokens.
            estimated = approx_tokens(json.dumps(messages, ensure_ascii=False))
            if estimated > 6000 and context:
                # Drop whole fragments from the end until the prompt is safe.
                blocks = context.split("\n\n")
                while blocks and approx_tokens(
                    json.dumps(prompt_with_context(question, "\n\n".join(blocks)), ensure_ascii=False)
                ) > 6000:
                    blocks.pop()
                context = "\n\n".join(blocks)
                messages = prompt_with_context(question, context)

            valid_citations = set(context_ids)
            llm_calls = 0
            input_tokens = 0
            output_tokens = 0
            answer: dict = {}
            last_content = ""

            for attempt in range(2):
                try:
                    content, in_tok, out_tok = call_llm(
                        client, model, messages, cfg, qid, submission_id
                    )
                except Exception as exc:  # network/API errors are reported in-band
                    answer = {
                        "answer_ids": [],
                        "abstain": True,
                        "rationale": f"Ошибка обращения к модели: {exc}"[:1200],
                        "citations": [],
                    }
                    llm_calls += 1
                    break

                llm_calls += 1
                input_tokens += in_tok or approx_tokens(
                    json.dumps(messages, ensure_ascii=False)
                )
                output_tokens += out_tok or approx_tokens(content)
                last_content = content
                answer = extract_answer(content, question, valid_citations, cfg)

                if answer and (
                    answer["abstain"]
                    or (answer["answer_ids"] and answer["citations"])
                    or args.mode == "no_context"
                ):
                    break
                if attempt == 0:
                    messages = repair_messages(messages, content)
                    estimated = approx_tokens(
                        json.dumps(messages, ensure_ascii=False)
                    )
                    if estimated > 6000:
                        # A repair prompt that is too large is not attempted.
                        break

            if not answer:
                # Last-resort recovery from malformed JSON.
                data = parse_json_object(last_content) or {}
                answer = extract_answer(
                    json.dumps(data, ensure_ascii=False),
                    question,
                    valid_citations,
                    cfg,
                ) if data else {}

            if not answer:
                answer = {
                    "answer_ids": [],
                    "abstain": True,
                    "rationale": "Не удалось получить корректный структурированный ответ.",
                    "citations": [],
                }

            if answer["abstain"]:
                answer["answer_ids"] = []
                answer["citations"] = []
            else:
                if not answer["answer_ids"]:
                    answer["abstain"] = True
                    answer["citations"] = []
                    answer["rationale"] = (
                        answer.get("rationale")
                        or "Модель не выбрала ни одного варианта."
                    )
                elif args.mode != "no_context" and not answer["citations"]:
                    # Never invent links, but the top-ranked fragments are the
                    # honest fallback when the model omitted citations.
                    answer["citations"] = context_ids[:2]
            fallback_rationale = (
                "Недостаточно данных для однозначного ответа."
                if answer["abstain"]
                else "Ответ основан на указанных фрагментах."
            )
            answer["rationale"] = normalize_rationale(
                answer.get("rationale") or fallback_rationale,
                cfg["prompt"]["max_rationale_chars"],
            ) or fallback_rationale

            record = {
                "id": qid,
                "answer_ids": answer["answer_ids"],
                "abstain": bool(answer["abstain"]),
                "rationale": answer["rationale"],
                "citations": answer["citations"][: cfg["prompt"]["max_citations"]],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            elapsed_ms = int(round((time.perf_counter() - started) * 1000))
            trace_f.write(
                json.dumps(
                    {
                        "id": qid,
                        "llm_calls": llm_calls,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "elapsed_ms": elapsed_ms,
                        "retrieved_block_ids": context_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()
            trace_f.flush()
            print(
                f"{qid}: {record['answer_ids'] or 'ABSTAIN'} "
                f"calls={llm_calls} in={input_tokens} out={output_tokens} "
                f"{elapsed_ms}ms",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
