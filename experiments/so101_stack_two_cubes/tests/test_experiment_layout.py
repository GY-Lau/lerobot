#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_DIRS = {"assets", "docs", "manifests", "results", "scripts", "tests"}
ALLOWED_ROOT_FILES = {"README.md", "PROJECT_REPORT.md", "PROJECT_REPORT.zh-CN.md"}


class ExperimentLayoutTest(unittest.TestCase):
    def test_root_contains_only_entry_documents_and_named_directories(self):
        root_files = {path.name for path in EXPERIMENT_DIR.iterdir() if path.is_file()}
        root_dirs = {path.name for path in EXPERIMENT_DIR.iterdir() if path.is_dir()}
        self.assertEqual(root_files, ALLOWED_ROOT_FILES)
        self.assertEqual(root_dirs, EXPECTED_DIRS)

    def test_local_markdown_links_resolve(self):
        missing: list[str] = []
        link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
        for document in EXPERIMENT_DIR.rglob("*.md"):
            for target in link_pattern.findall(document.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "#")):
                    continue
                path_text = target.split("#", 1)[0]
                if path_text and not (document.parent / path_text).resolve().exists():
                    missing.append(f"{document.relative_to(EXPERIMENT_DIR)} -> {target}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
