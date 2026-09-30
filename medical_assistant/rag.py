"""Offline document ingestion and lightweight BM25 + vector-style RRF retrieval.

The implementation deliberately has no network dependency. If ``pypdf`` or
``jieba`` is installed they are used; otherwise text files and character/word
tokens still provide a useful local search experience.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import SearchResult

try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    jieba = None


def tokenize(text: str) -> list[str]:
    text = text.lower()
    if jieba:
        words = [word.strip() for word in jieba.lcut(text) if word.strip()]
    else:
        words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return [word for word in words if len(word) > 1 or "\u4e00" <= word <= "\u9fff"]


class DocumentStore:
    def __init__(self, index_path: str | Path, chunk_size: int = 420, chunk_overlap: int = 60):
        self.index_path = Path(index_path)
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.documents: list[dict] = []
        self.load()

    def load(self) -> None:
        try:
            self.documents = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(self.documents, list):
                self.documents = []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.documents = []

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")

    def ingest_text(self, title: str, text: str, path: str = "") -> int:
        text = re.sub(r"\r\n?", "\n", text).strip()
        if not text:
            return 0
        # Remove older chunks from the same source so re-import is idempotent.
        if path:
            self.documents = [item for item in self.documents if item.get("path") != path]
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = 0
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size].strip()
            if not piece:
                continue
            self.documents.append({"chunk_id": uuid.uuid4().hex[:12], "title": title, "path": path, "text": piece})
            chunks += 1
            if start + self.chunk_size >= len(text):
                break
        self.save()
        return chunks

    def ingest_file(self, path: str | Path) -> int:
        file_path = Path(path)
        if file_path.suffix.lower() == ".pdf":
            text = self._read_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(file_path.stem, text, str(file_path))

    def ingest_directory(self, directory: str | Path) -> int:
        total = 0
        for path in sorted(Path(directory).glob("**/*")):
            if path.suffix.lower() in {".txt", ".md", ".pdf"} and path.is_file():
                try:
                    total += self.ingest_file(path)
                except OSError:
                    continue
        return total

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            return ""
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        except Exception:
            return ""

    def search(self, query: str, top_k: int = 5, rrf_weight: float = 0.65) -> list[SearchResult]:
        if not query.strip() or not self.documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        corpus_tokens = [tokenize(item["text"]) for item in self.documents]
        df = Counter(token for tokens in corpus_tokens for token in set(tokens))
        n = len(corpus_tokens)
        avg_len = sum(len(tokens) for tokens in corpus_tokens) / max(1, n)
        lexical: list[tuple[int, float]] = []
        vector: list[tuple[int, float]] = []
        query_counts = Counter(query_tokens)
        for idx, tokens in enumerate(corpus_tokens):
            counts = Counter(tokens)
            score = 0.0
            for token, qtf in query_counts.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * (tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(tokens) / max(1, avg_len)))) * qtf
            lexical.append((idx, score))
            # A sparse cosine score acts as a dependency-free local embedding fallback.
            dot = sum(counts[t] * query_counts[t] for t in query_counts)
            denom = math.sqrt(sum(v * v for v in counts.values()) * sum(v * v for v in query_counts.values()))
            vector.append((idx, dot / denom if denom else 0.0))
        lexical.sort(key=lambda item: item[1], reverse=True)
        vector.sort(key=lambda item: item[1], reverse=True)
        rank_lex = {idx: rank for rank, (idx, _) in enumerate(lexical)}
        rank_vec = {idx: rank for rank, (idx, _) in enumerate(vector)}
        rrf: list[tuple[int, float]] = []
        for idx in range(n):
            score = rrf_weight / (60 + rank_lex.get(idx, n)) + (1 - rrf_weight) / (60 + rank_vec.get(idx, n))
            if lexical[idx][1] > 0 or vector[idx][1] > 0:
                rrf.append((idx, score))
        rrf.sort(key=lambda item: item[1], reverse=True)
        return [SearchResult(self.documents[idx]["title"], self.documents[idx]["text"], score, self.documents[idx].get("path", ""), self.documents[idx].get("chunk_id", "")) for idx, score in rrf[: max(1, top_k)]]
