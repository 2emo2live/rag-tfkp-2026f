# download_models.py
"""One-time setup: download embedding/reranker weights before the run."""
import json, sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

cfg = json.loads(Path("config.json").read_text())
for key in ("embedding", "reranker"):
    m = cfg.get(key, {})
    name = m.get("model")
    if not name:
        continue
    rev = m.get("revision")
    print(f"downloading {key}: {name}@{rev}")
    SentenceTransformer(name, revision=rev, device="cpu")
    print("  done")