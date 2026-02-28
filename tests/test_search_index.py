import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from search_index import SearchIndex, SearchIndexBuilder


def _supports_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


class SearchIndexTests(unittest.TestCase):
    @unittest.skipUnless(_supports_fts5(), "SQLite FTS5 not available")
    def test_build_and_search_keyword_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            md_path = output_dir / "2024-01-01_Test_Search.md"
            md_path.write_text(
                "\n".join(
                    [
                        "# Test Search",
                        "",
                        "**Created:** 2024-01-01 12:00:00  ",
                        "**Updated:** 2024-01-01 12:05:00  ",
                        "**Conversation ID:** `abc123`",
                        "",
                        "---",
                        "",
                        "### **User**",
                        "",
                        "We should cache vector embeddings to speed retrieval.",
                        "",
                        "### **Assistant**",
                        "",
                        "Agreed. Add a cache layer for embeddings and reuse results.",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"CHATGPT_HISTORY_EMBEDDING_BACKEND": "off"}):
                SearchIndexBuilder(output_dir).build()

            index = SearchIndex.try_load(output_dir)
            self.assertIsNotNone(index)

            results = index.search("vector cache", limit=5, mode="keyword")
            self.assertTrue(results)
            self.assertEqual(results[0]["file"], md_path.name)
            self.assertIn("snippet", results[0])

    @unittest.skipUnless(_supports_fts5(), "SQLite FTS5 not available")
    def test_title_scope_and_role_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            md_path = output_dir / "2024-01-01_Title_Only.md"
            md_path.write_text(
                "\n".join(
                    [
                        "# Unique Topic Title",
                        "",
                        "**Created:** 2024-01-01 09:00:00  ",
                        "**Updated:** 2024-01-01 09:05:00  ",
                        "**Conversation ID:** `xyz987`",
                        "",
                        "---",
                        "",
                        "### **User**",
                        "",
                        "General update.",
                        "",
                        "### **Assistant**",
                        "",
                        "Response text.",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"CHATGPT_HISTORY_EMBEDDING_BACKEND": "off"}):
                SearchIndexBuilder(output_dir).build()

            index = SearchIndex.try_load(output_dir)
            self.assertIsNotNone(index)

            results = index.search(
                "Unique Topic",
                limit=5,
                scope="title",
                role="assistant",
                mode="keyword",
            )
            self.assertTrue(results)
            self.assertEqual(results[0]["file"], md_path.name)

    @unittest.skipUnless(_supports_fts5(), "SQLite FTS5 not available")
    def test_sort_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            older = output_dir / "2024-01-01_Old.md"
            newer = output_dir / "2024-02-01_New.md"

            older.write_text(
                "\n".join(
                    [
                        "# Sort Test",
                        "",
                        "**Created:** 2024-01-01 08:00:00  ",
                        "**Updated:** 2024-01-01 08:05:00  ",
                        "**Conversation ID:** `old`",
                        "",
                        "---",
                        "",
                        "### **User**",
                        "",
                        "Search phrase for sorting.",
                    ]
                ),
                encoding="utf-8",
            )
            newer.write_text(
                "\n".join(
                    [
                        "# Sort Test",
                        "",
                        "**Created:** 2024-02-01 08:00:00  ",
                        "**Updated:** 2024-02-01 08:05:00  ",
                        "**Conversation ID:** `new`",
                        "",
                        "---",
                        "",
                        "### **User**",
                        "",
                        "Search phrase for sorting.",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"CHATGPT_HISTORY_EMBEDDING_BACKEND": "off"}):
                SearchIndexBuilder(output_dir).build()

            index = SearchIndex.try_load(output_dir)
            self.assertIsNotNone(index)

            results = index.search(
                "Search phrase",
                limit=2,
                sort="newest",
                mode="keyword",
            )
            self.assertEqual(results[0]["file"], newer.name)

    def test_empty_query_returns_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            md_path = output_dir / "2024-01-01_Empty.md"
            md_path.write_text(
                "\n".join(
                    [
                        "# Empty",
                        "",
                        "**Created:** 2024-01-01 00:00:00  ",
                        "**Updated:** 2024-01-01 00:00:00  ",
                        "**Conversation ID:** `none`",
                        "",
                        "---",
                        "",
                        "### **User**",
                        "",
                        "Hello world.",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"CHATGPT_HISTORY_EMBEDDING_BACKEND": "off"}):
                SearchIndexBuilder(output_dir).build()

            index = SearchIndex.try_load(output_dir)
            self.assertIsNotNone(index)
            self.assertEqual(index.search("", limit=5), [])
