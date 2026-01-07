#!/usr/bin/env python3
"""
Search index builder and query utilities for the ChatGPT Markdown archive.

Hybrid retrieval uses:
- Keyword search via SQLite FTS5
- Optional vector search (hashing or sentence-transformers embeddings)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import struct
import threading
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

INDEX_DB_NAME = "search.sqlite"
META_JSON_NAME = "search_meta.json"
EMBEDDINGS_BIN_NAME = "search_embeddings.bin"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_HASH_DIM = 384


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _safe_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text.lower())


def _chunk_text(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + chunk_words)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(tokens):
            break
    return chunks


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        inv = 1.0 / norm
        for i in range(len(vec)):
            vec[i] *= inv
    return vec


def _fts_query(query: str, scope: str) -> str:
    tokens = _tokenize(query)
    if not tokens:
        return ""
    if scope == "title":
        return " ".join([f"title:{token}" for token in tokens])
    if scope == "content":
        return " ".join([f"content:{token}" for token in tokens])
    return " ".join(tokens)


def _parse_datetime(value: str, end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value or value.lower() == "unknown":
        return None
    if len(value) == 10:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None
        if end_of_day:
            return parsed.replace(hour=23, minute=59, second=59)
        return parsed
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_timestamp(value: str) -> Optional[float]:
    parsed = _parse_datetime(value)
    return parsed.timestamp() if parsed else None


def _parse_role(header_line: str) -> str:
    match = re.search(r"\*\*(User|Assistant|System|Tool)\*\*", header_line, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "unknown"


def _parse_metadata(text: str, fallback_title: str) -> Dict[str, str]:
    title = fallback_title
    created = ""
    updated = ""
    conversation_id = ""
    for line in text.splitlines()[:30]:
        if line.startswith("# "):
            title = line[2:].strip() or title
        elif line.startswith("**Created:**"):
            created = line.split("**Created:**", 1)[1].strip()
        elif line.startswith("**Updated:**"):
            updated = line.split("**Updated:**", 1)[1].strip()
        elif line.startswith("**Conversation ID:**"):
            conversation_id = line.split("**Conversation ID:**", 1)[1].strip(" `")
    return {
        "title": title,
        "created": created,
        "updated": updated,
        "conversation_id": conversation_id,
    }


def _is_message_header(line: str) -> bool:
    if not line.startswith("### "):
        return False
    return bool(re.search(r"\*\*(User|Assistant|System|Tool)\*\*", line, re.IGNORECASE))


def _extract_message_blocks(text: str) -> Iterable[Tuple[int, str, str]]:
    lines = text.splitlines()
    body_start = 0
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            body_start = idx + 1
            break

    lines = lines[body_start:]
    messages: List[Tuple[int, str, str]] = []
    current_role = "unknown"
    current_lines: List[str] = []
    message_index = 1

    def flush() -> None:
        nonlocal message_index, current_lines
        content = _normalize_whitespace("\n".join(current_lines))
        if content:
            messages.append((message_index, current_role, content))
            message_index += 1
        current_lines = []

    for line in lines:
        if _is_message_header(line):
            flush()
            current_role = _parse_role(line)
            continue
        if line.startswith("![Attachment](") or line.startswith("📎 [Attachment]("):
            continue
        current_lines.append(line)

    flush()
    return messages


class HashingEmbedder:
    def __init__(self, dim: int = DEFAULT_HASH_DIM) -> None:
        self.dim = dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _tokenize(text):
                digest = struct.unpack("<I", _hash_token(token))[0]
                idx = digest % self.dim
                sign = 1.0 if (digest & 0x1) else -1.0
                vec[idx] += sign
            embeddings.append(_l2_normalize(vec))
        return embeddings


def _hash_token(token: str) -> bytes:
    import hashlib

    return hashlib.md5(token.encode("utf-8")).digest()[:4]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def _resolve_embedder(
    backend: str, model_name: str, dim: int, *, allow_fallback: bool
) -> Tuple[str, Optional[object], int]:
    backend = (backend or "auto").strip().lower()

    if backend in ("off", "none", "disable", "disabled"):
        return "off", None, 0

    if backend in ("auto", ""):
        backend = "sentence-transformers"

    if backend in ("sentence-transformers", "sbert", "sentence_transformers"):
        try:
            return "sentence-transformers", SentenceTransformerEmbedder(model_name), 0
        except Exception as exc:
            if allow_fallback:
                logger.warning(
                    "Sentence-transformers unavailable (%s); falling back to hashing.",
                    exc,
                )
                return "hash", HashingEmbedder(dim), dim
            logger.warning("Sentence-transformers unavailable (%s); disabling vector search.", exc)
            return "off", None, 0

    if backend in ("hash", "hashing"):
        return "hash", HashingEmbedder(dim), dim

    logger.warning("Unknown embedding backend %r; disabling embeddings.", backend)
    return "off", None, 0


@dataclass
class SearchChunk:
    rowid: int
    content: str
    title: str
    file: str
    role: str
    created: str
    conversation_id: str
    message_index: int
    chunk_index: int


class SearchIndexBuilder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / INDEX_DB_NAME
        self.meta_path = self.output_dir / META_JSON_NAME
        self.embeddings_path = self.output_dir / EMBEDDINGS_BIN_NAME

        self.chunk_words = _safe_int(os.environ.get("CHATGPT_HISTORY_CHUNK_WORDS"), 320)
        self.chunk_overlap = _safe_int(os.environ.get("CHATGPT_HISTORY_CHUNK_OVERLAP"), 64)

        backend_env = os.environ.get("CHATGPT_HISTORY_EMBEDDING_BACKEND", "auto")
        model_env = os.environ.get("CHATGPT_HISTORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        dim_env = _safe_int(os.environ.get("CHATGPT_HISTORY_HASH_DIM"), DEFAULT_HASH_DIM)

        backend, embedder, dim = _resolve_embedder(
            backend_env, model_env, dim_env, allow_fallback=True
        )
        self.embedding_backend = backend
        self.embedding_model = model_env
        self.embedding_dim = dim
        self.embedder = embedder

    def build(self) -> None:
        markdown_files = sorted(self.output_dir.glob("*.md"))
        markdown_files = [path for path in markdown_files if path.name != "index.md"]

        if not markdown_files:
            logger.warning("No markdown files found to index in %s", self.output_dir)
            return

        if self.index_path.exists():
            self.index_path.unlink()

        logger.info("Building search index (%s files)...", len(markdown_files))
        conn = sqlite3.connect(str(self.index_path))
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE chunks USING fts5(
                    content,
                    title,
                    file UNINDEXED,
                    role UNINDEXED,
                    created UNINDEXED,
                    conversation_id UNINDEXED,
                    message_index UNINDEXED,
                    chunk_index UNINDEXED
                )
                """
            )
        except sqlite3.OperationalError as exc:
            conn.close()
            logger.warning("Failed to create FTS5 index: %s", exc)
            return

        rows: List[Tuple] = []
        meta_items: List[Dict[str, object]] = []
        embeddings: List[List[float]] = []
        batch_texts: List[str] = []

        rowid = 0

        def flush_rows() -> None:
            if not rows:
                return
            conn.executemany(
                """
                INSERT INTO chunks (
                    rowid, content, title, file, role, created, conversation_id, message_index, chunk_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            rows.clear()

        def flush_embeddings() -> None:
            if not batch_texts or not self.embedder:
                batch_texts.clear()
                return
            try:
                batch_embeddings = self.embedder.encode(batch_texts)
            except Exception as exc:
                logger.warning("Embedding batch failed (%s); continuing keyword-only.", exc)
                self.embedder = None
                self.embedding_backend = "off"
                embeddings.clear()
                batch_texts.clear()
                return
            embeddings.extend(batch_embeddings)
            batch_texts.clear()

        for path in markdown_files:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
                continue

            metadata = _parse_metadata(content, path.stem)
            messages = _extract_message_blocks(content)

            for message_index, role, message_text in messages:
                for chunk_index, chunk in enumerate(
                    _chunk_text(message_text, self.chunk_words, self.chunk_overlap), 1
                ):
                    rowid += 1
                    rows.append(
                        (
                            rowid,
                            chunk,
                            metadata["title"],
                            path.name,
                            role,
                            metadata["created"],
                            metadata["conversation_id"],
                            message_index,
                            chunk_index,
                        )
                    )
                    meta_items.append(
                        {
                            "rowid": rowid,
                            "title": metadata["title"],
                            "file": path.name,
                            "role": role,
                            "created": metadata["created"],
                            "conversation_id": metadata["conversation_id"],
                            "message_index": message_index,
                            "chunk_index": chunk_index,
                        }
                    )

                    if self.embedder:
                        batch_texts.append(chunk)

                    if len(rows) >= 500:
                        flush_rows()

                    if self.embedder and len(batch_texts) >= 64:
                        flush_embeddings()

        flush_rows()
        flush_embeddings()
        conn.commit()
        conn.close()

        embedding_dim = len(embeddings[0]) if embeddings else 0
        payload = {
            "backend": self.embedding_backend,
            "model": self.embedding_model,
            "embedding_dim": embedding_dim,
            "chunk_words": self.chunk_words,
            "chunk_overlap": self.chunk_overlap,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": meta_items,
        }
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if self.embedder and embeddings:
            self._write_embeddings(embeddings)
            logger.info(
                "Search index ready (%d chunks, vector backend=%s).",
                len(meta_items),
                self.embedding_backend,
            )
        else:
            if self.embeddings_path.exists():
                self.embeddings_path.unlink()
            logger.info("Search index ready (%d chunks, keyword-only).", len(meta_items))

    def _write_embeddings(self, embeddings: List[List[float]]) -> None:
        if not embeddings:
            return
        dim = len(embeddings[0])
        count = len(embeddings)
        with open(self.embeddings_path, "wb") as handle:
            handle.write(b"CHIDX1")
            handle.write(struct.pack("<II", dim, count))
            flat = array("f")
            for vec in embeddings:
                flat.extend(vec)
            flat.tofile(handle)


class SearchIndex:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / INDEX_DB_NAME
        self.meta_path = self.output_dir / META_JSON_NAME
        self.embeddings_path = self.output_dir / EMBEDDINGS_BIN_NAME

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing search index at {self.index_path}")

        self._conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

        meta_payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._meta_items = meta_payload.get("items", [])
        self._meta_by_rowid = {item["rowid"]: item for item in self._meta_items}
        backend = meta_payload.get("backend", "off")
        model = meta_payload.get("model", DEFAULT_EMBEDDING_MODEL)
        embedding_dim = meta_payload.get("embedding_dim", 0)

        self._embedding_backend = backend
        self._embedding_model = model
        self._embedding_dim = embedding_dim

        self._embedder: Optional[object] = None
        self._vectors: Optional[array] = None
        self._vector_count = 0
        self._vector_scan_limit = max(
            0, _safe_int(os.environ.get("CHATGPT_HISTORY_VECTOR_SCAN_LIMIT"), 20000)
        )
        self._created_ts_by_rowid = {
            int(item["rowid"]): _parse_timestamp(str(item.get("created", "")))
            for item in self._meta_items
            if "rowid" in item
        }

        if self.embeddings_path.exists() and backend != "off":
            _, embedder, dim = _resolve_embedder(
                backend, model, embedding_dim, allow_fallback=False
            )
            self._embedder = embedder
            if self._embedder:
                self._load_embeddings(dim)

    @classmethod
    def try_load(cls, output_dir: Path) -> Optional["SearchIndex"]:
        try:
            return cls(output_dir)
        except Exception as exc:
            logger.warning("Search index unavailable: %s", exc)
            return None

    def _load_embeddings(self, dim: int) -> None:
        with open(self.embeddings_path, "rb") as handle:
            magic = handle.read(6)
            if magic != b"CHIDX1":
                logger.warning("Unsupported embeddings file format.")
                return
            stored_dim, count = struct.unpack("<II", handle.read(8))
            if dim and stored_dim != dim:
                logger.warning("Embedding dimension mismatch (expected %s, got %s).", dim, stored_dim)
                return
            vectors = array("f")
            vectors.fromfile(handle, count * stored_dim)
            if count != len(self._meta_items):
                logger.warning("Embedding count mismatch (expected %s, got %s).", len(self._meta_items), count)
                return
            self._embedding_dim = stored_dim
            self._vector_count = count
            self._vectors = vectors

    def search(
        self,
        query: str,
        limit: int = 20,
        mode: str = "hybrid",
        scope: str = "both",
        role: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort: str = "relevance",
        snippet_tokens: int = 12,
    ) -> List[Dict[str, object]]:
        if not query.strip():
            return []

        mode = (mode or "hybrid").strip().lower()
        scope = (scope or "both").strip().lower()
        sort = (sort or "relevance").strip().lower()

        if scope not in ("both", "content", "title"):
            scope = "both"
        if mode not in ("hybrid", "keyword", "vector"):
            mode = "hybrid"
        if sort not in ("relevance", "newest"):
            sort = "relevance"

        if scope == "title" and mode == "vector":
            mode = "keyword"

        role_filter = (role or "").strip().lower() or None
        date_from_dt = _parse_datetime(date_from or "")
        date_to_dt = _parse_datetime(date_to or "", end_of_day=True)
        date_from_ts = date_from_dt.timestamp() if date_from_dt else None
        date_to_ts = date_to_dt.timestamp() if date_to_dt else None

        snippet_tokens = max(6, min(int(snippet_tokens or 12), 40))

        needs_filtering = role_filter or date_from_ts or date_to_ts
        base_limit = limit * (6 if needs_filtering else 3)

        keyword_hits: List[Dict[str, object]] = []
        vector_hits: List[Dict[str, object]] = []

        if mode in ("hybrid", "keyword"):
            keyword_hits = self._keyword_search(
                query, limit=base_limit, scope=scope, snippet_tokens=snippet_tokens
            )
            keyword_hits = self._filter_hits(
                keyword_hits, role_filter, date_from_ts, date_to_ts
            )

        if mode in ("hybrid", "vector"):
            candidate_rowids = [hit["rowid"] for hit in keyword_hits]
            vector_hits = self._vector_search(
                query,
                limit=base_limit,
                candidate_rowids=candidate_rowids,
                snippet_tokens=snippet_tokens,
                scope=scope,
            )
            vector_hits = self._filter_hits(
                vector_hits, role_filter, date_from_ts, date_to_ts
            )

        scores: Dict[int, float] = {}
        results: Dict[int, Dict[str, object]] = {}

        def add_score(rowid: int, score: float) -> None:
            scores[rowid] = scores.get(rowid, 0.0) + score

        if mode == "keyword":
            for rank, hit in enumerate(keyword_hits, 1):
                rowid = hit["rowid"]
                add_score(rowid, hit.get("keyword_score", 0.0))
                results[rowid] = hit
        elif mode == "vector":
            for rank, hit in enumerate(vector_hits, 1):
                rowid = hit["rowid"]
                add_score(rowid, hit.get("vector_score", 0.0))
                results[rowid] = hit
        else:
            for rank, hit in enumerate(keyword_hits, 1):
                rowid = hit["rowid"]
                add_score(rowid, 0.6 / (rank + 60))
                results[rowid] = hit

            for rank, hit in enumerate(vector_hits, 1):
                rowid = hit["rowid"]
                add_score(rowid, 0.4 / (rank + 60))
                if rowid not in results:
                    results[rowid] = hit

        merged = [{**results[rowid], "score": scores[rowid]} for rowid in scores.keys()]
        if sort == "newest":
            merged.sort(
                key=lambda item: (
                    self._created_ts_by_rowid.get(int(item["rowid"]), 0) or 0,
                    item.get("score", 0.0),
                ),
                reverse=True,
            )
        else:
            merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:limit]

    def _keyword_search(
        self, query: str, limit: int, scope: str, snippet_tokens: int
    ) -> List[Dict[str, object]]:
        fts = _fts_query(query, scope)
        if not fts:
            return []
        snippet_col = 1 if scope == "title" else 0
        snippet_tokens = max(6, min(int(snippet_tokens or 12), 40))
        snippet_sql = (
            f"snippet(chunks, {snippet_col}, '[[', ']]', '...', {snippet_tokens})"
        )
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT
                    rowid,
                    title,
                    file,
                    role,
                    created,
                    conversation_id,
                    message_index,
                    chunk_index,
                    {snippet_sql} AS snippet,
                    bm25(chunks) AS rank
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts, limit),
            ).fetchall()

        results = []
        for row in rows:
            rank = float(row["rank"])
            score = 1.0 / (1.0 + rank) if rank >= 0 else 1.0
            snippet = row["snippet"]
            if scope == "both" and not snippet:
                snippet = row["title"]
            results.append(
                {
                    "rowid": row["rowid"],
                    "title": row["title"],
                    "file": row["file"],
                    "role": row["role"],
                    "created": row["created"],
                    "conversation_id": row["conversation_id"],
                    "message_index": row["message_index"],
                    "chunk_index": row["chunk_index"],
                    "snippet": snippet,
                    "keyword_score": score,
                }
            )
        return results

    def _vector_search(
        self,
        query: str,
        limit: int,
        candidate_rowids: Optional[List[int]] = None,
        snippet_tokens: int = 12,
        scope: str = "both",
    ) -> List[Dict[str, object]]:
        if not self._embedder or not self._vectors or not self._vector_count:
            return []

        query_vec = self._embedder.encode([query])[0]
        query_vec = _l2_normalize(list(query_vec))

        dim = self._embedding_dim
        vectors = self._vectors
        hits: List[Tuple[float, int]] = []

        if candidate_rowids:
            indices = sorted(
                {
                    rowid - 1
                    for rowid in candidate_rowids
                    if isinstance(rowid, int) and 1 <= rowid <= self._vector_count
                }
            )
        else:
            scan_limit = self._vector_scan_limit
            if scan_limit <= 0 or scan_limit > self._vector_count:
                scan_limit = self._vector_count
            indices = range(scan_limit)

        for i in indices:
            offset = i * dim
            score = 0.0
            for j in range(dim):
                score += vectors[offset + j] * query_vec[j]
            hits.append((score, i))

        hits.sort(key=lambda item: item[0], reverse=True)
        top_hits = hits[:limit]

        results: List[Dict[str, object]] = []
        for score, idx in top_hits:
            if idx >= len(self._meta_items):
                continue
            meta = self._meta_items[idx]
            rowid = int(meta["rowid"])
            snippet = self._fetch_snippet(rowid, query, snippet_tokens, scope)
            if not snippet:
                snippet = self._fetch_content(rowid)
            results.append(
                {
                    "rowid": rowid,
                    "title": meta.get("title", ""),
                    "file": meta.get("file", ""),
                    "role": meta.get("role", ""),
                    "created": meta.get("created", ""),
                    "conversation_id": meta.get("conversation_id", ""),
                    "message_index": meta.get("message_index", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "snippet": snippet,
                    "vector_score": score,
                }
            )
        return results

    def _fetch_snippet(
        self, rowid: int, query: str, snippet_tokens: int, scope: str
    ) -> str:
        fts = _fts_query(query, scope)
        if not fts:
            return ""
        snippet_col = 1 if scope == "title" else 0
        snippet_tokens = max(6, min(int(snippet_tokens or 12), 40))
        snippet_sql = (
            f"snippet(chunks, {snippet_col}, '[[', ']]', '...', {snippet_tokens})"
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT {snippet_sql} AS snippet
                FROM chunks
                WHERE rowid = ? AND chunks MATCH ?
                """,
                (rowid, fts),
            ).fetchone()
        if row and row["snippet"]:
            return row["snippet"]
        return ""

    def _fetch_content(self, rowid: int) -> str:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT content
                FROM chunks
                WHERE rowid = ?
                """,
                (rowid,),
            ).fetchone()
        if row and row["content"]:
            content = row["content"].strip()
            if len(content) > 240:
                return content[:237].rstrip() + "..."
            return content
        return ""

    def _filter_hits(
        self,
        hits: List[Dict[str, object]],
        role: Optional[str],
        date_from: Optional[float],
        date_to: Optional[float],
    ) -> List[Dict[str, object]]:
        if not hits:
            return []
        if not role and date_from is None and date_to is None:
            return hits

        filtered: List[Dict[str, object]] = []
        for hit in hits:
            rowid = int(hit["rowid"])
            if role and str(hit.get("role", "")).lower() != role:
                continue
            created_ts = self._created_ts_by_rowid.get(rowid)
            if date_from is not None:
                if created_ts is None or created_ts < date_from:
                    continue
            if date_to is not None:
                if created_ts is None or created_ts > date_to:
                    continue
            filtered.append(hit)
        return filtered
