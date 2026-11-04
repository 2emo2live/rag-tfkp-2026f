# Доказательный RAG по ТФКП

**Автор:** Семенов Иван Алексеевич  
**Группа:** М05-512д

Система отвечает на вопросы с выбором вариантов по учебнику «Лекции по ТФКП» (Горяйнов, Половинкин, 2025), приводит короткое обоснование и ссылки на фрагменты корпуса. Если доказательств в найденных фрагментах недостаточно, система возвращает отказ.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="mlx-community/Qwen3.5-4B-MLX-4bit"
export LLM_API_KEY="local"
export SUBMISSION_ID="local-dev"

python build_index.py \
  --corpus /path/to/corpus.jsonl \
  --output ./index \
  --config config.json

python predict.py \
  --index ./index \
  --questions /path/to/questions.jsonl \
  --output answers.jsonl \
  --trace trace.jsonl \
  --config config.json
```

`predict.py` поддерживает режимы `final`, `rag_single`, `rag_bm25_dual`, `no_context`, `no_hybrid`, `no_reranker`. `rag_bm25_dual` использует текстовый BM25 и резервирует один слот контекста для формульного BM25. Режим по умолчанию — `final`.

## Формат корпуса

Каждая строка `corpus.jsonl`:

```json
{"block_id": "b_p011_001", "page": 11, "section": "§ 2. ...", "text": "..."}
```

Корпус и индекс **не входят в репозиторий**: индекс пересобирается из корпуса, кода и публичных моделей. В рабочей папке корпус собран из PDF скриптом `make_corpus.py`; `build_index.py` принимает и официальный `corpus.jsonl` в том же формате.

## Архитектура

### Корпус и чанкинг

- 360 чанков, страницы PDF 3–147, средняя длина около 754 символов.
- Чанки page-local, `block_id` вида `b_pNNN_NNN`; цитаты ссылаются на эти ID.
- Числители и знаменатели формул, которые PDF отдаёт отдельными строками, восстанавливаются: убрано правило, считавшее любую цифровую строку колонтитулом.
- Разорванные слова переносятся в следующий чанк без изменения числа чанков и ID.

### Поиск

- BM25 по леммам `pymorphy3`, `k1=1.5`, `b=0.75`.
- Dense-поиск: `intfloat/multilingual-e5-base`, 768-мерные нормализованные эмбеддинги, префиксы `query:` / `passage:`.
- Гибрид: RRF с `k=60`, по 40 кандидатов от BM25 и dense.
- Реранкер: `DiTy/cross-encoder-russian-msmarco`, до 24 кандидатов, CPU.
- В контекст попадают top-6 после реранкера; для top-4 добавляются соседние чанки. Лимит контекста — 9000 символов.

### Генерация и отказ

- Qwen3.5-4B, thinking выключен, `temperature=0.0`, `top_p=1.0`, `max_tokens=900`, `seed=42`.
- Ответ — JSON: `answer_ids`, `abstain`, `rationale`, `citations`.
- Модель обязана цитировать только `block_id` из переданного контекста; лишние и неизвестные ID отбрасываются.
- При недостатке данных возвращается `abstain=true` с пустыми `answer_ids` и `citations`.
- На вопрос допускается не более 4 вызовов LLM; фактически используется 1–2.

## Ресурсы

| Параметр | Значение |
|---|---:|
| Оборудование | Apple M4, 16 ГБ RAM, macOS 15.2 |
| Сборка корпуса | 1.0 с, пик RSS 66 МБ |
| Сборка индекса | 35.3 с, пик RSS 1.39 ГиБ |
| Размер индекса | 2.0 МБ (2 109 454 байта) |
| Пик RSS `predict.py` | около 2.0 ГиБ |
| vLLM EngineCore | около 3.5 ГБ во время инференса |

## Эксперименты

Таблица результатов на `dev_questions.jsonl`:

| Режим | Answer accuracy | Q | Ср. input | Ср. output | Ср. вызовов | Ср. стоимость | Медиана latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_context` | см. `experiments.csv` | — | — | — | — | — | — |
| `rag_single` | см. `experiments.csv` | — | — | — | — | — | — |
| `final` | см. `experiments.csv` | — | — | — | — | — | — |
| `ablation_no_hybrid` | см. `experiments.csv` | — | — | — | — | — | — |
| `ablation_no_reranker` | см. `experiments.csv` | — | — | — | — | — | — |

Актуальные числа — в `experiments.csv`; выводы по абляциям — в `report.md`.

## Файлы

- `build_index.py` — сборка BM25/dense индекса.
- `predict.py` — поиск, контекст, вызов LLM, ответ и trace.
- `evaluate.py` — метрика задания.
- `check_format.py` — проверка формата ответов и trace.
- `run_experiments.py` — пять режимов и `experiments.csv`.
- `make_corpus.py`, `make_dev.py`, `make_own.py` — вспомогательные скрипты подготовки данных.
- `config.json` — модели, ревизии, параметры поиска и генерации.
- `runtime.json` — доступ, sampling, квантизация, RAM, таймауты.
- `dev_answers.jsonl`, `dev_trace.jsonl` — итоговый прогон открытого набора.
- `own_questions.jsonl`, `own_gold.jsonl` — 8 авторских вопросов.

## Ограничения

- DiTy обучен на общих русскоязычных парах MS MARCO и на формульном корпусе может опускать нужные математические фрагменты; это видно в абляции `no_reranker` и разобрано в отчёте.
- Цитаты валидны в пределах переданного корпуса; при замене корпуса ID должны совпадать с его `block_id`.
- Система не использует внешние API и генеративные модели, кроме заданной Qwen.

## Внешний код и помощь ИИ

Использованы открытые библиотеки: PyMuPDF, pymorphy3, rank-bm25, sentence-transformers, transformers, PyTorch, OpenAI Python client. Код, отчёт и проверки подготовлены с помощью AI-ассистента OpenCode (DeepSeek V4.1 Flash); все эксперименты выполнены локально на указанном оборудовании.
