import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from converter import ChatGPTConverter


class ConverterIndexTests(unittest.TestCase):
    def test_create_index_handles_missing_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            converter = ChatGPTConverter(output_dir=str(output_dir))
            filenames = [
                ("Known Title", "2024-01-01_Known.md", datetime(2024, 1, 1, 12, 0, 0)),
                ("Unknown Title", "Unknown_Unknown.md", None),
            ]

            converter.create_index(filenames)

            index_path = output_dir / "index.md"
            content = index_path.read_text(encoding="utf-8")
            self.assertIn("## January 2024", content)
            self.assertIn("## Unknown Date", content)
            self.assertIn("[Known Title](2024-01-01_Known.md)", content)
            self.assertIn("[Unknown Title](Unknown_Unknown.md)", content)
