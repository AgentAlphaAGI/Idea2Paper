"""Synthetic regressions for endpoint and completion failures in extraction."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tools"
    / "extract_paper_review.py"
)
SPEC = importlib.util.spec_from_file_location("extract_paper_review", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


class ExtractPaperReviewFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        base_dir = Path(self.temp_dir.name) / "synthetic_fixture"
        paper_dir = base_dir / "synthetic_fixture_paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "case001_paper.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "title": "Synthetic fixture: two-stage puzzle solver",
                        "sections": [
                            {
                                "heading": "Abstract",
                                "text": "Synthetic input for parser regression only.",
                            },
                            {
                                "heading": "Method",
                                "text": "A toy deterministic transformation.",
                            },
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        self.base_dir = base_dir
        self.output_dir = (
            Path(self.temp_dir.name) / "extract_result" / "synthetic_fixture"
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def _run_batch(self):
        with mock.patch.object(
            sys, "argv", [str(SCRIPT_PATH), str(self.base_dir), "1"]
        ):
            return extractor.main()

    def test_invalid_mock_completion_fails_without_writing_paper_node(self):
        with mock.patch.object(
            extractor, "call_llm", return_value="not JSON: synthetic"
        ):
            exit_code = self._run_batch()

        self.assertEqual(exit_code, 1)
        failed_ids = json.loads(
            (self.output_dir / "_failed_paper_ids.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failed_ids, ["case001"])
        self.assertFalse((self.output_dir / "case001_paper_node.json").exists())
        self.assertFalse((self.output_dir / "_all_paper_nodes.json").exists())

    def test_valid_mock_completion_still_saves_node(self):
        mock_responses = [
            (
                '{"research_object":"synthetic",'
                '"core_technique":"mock","application":"mock",'
                '"domains":["synthetic"]}'
            ),
            (
                '{"core_idea":"synthetic","tech_stack":["mock"],'
                '"input_type":"synthetic","output_type":"synthetic"}'
            ),
            (
                '{"problem_framing":"mock","gap_pattern":"mock",'
                '"method_story":"mock","experiments_story":"mock"}'
            ),
            (
                '[{"name":"mock","type":"writing-level",'
                '"purpose":"test","location":"introduction",'
                '"description":"synthetic"}]'
            ),
        ]
        with mock.patch.object(extractor, "call_llm", side_effect=mock_responses):
            exit_code = self._run_batch()

        self.assertEqual(exit_code, 0)
        output = self.output_dir / "case001_paper_node.json"
        self.assertTrue(output.exists())
        failed_ids = json.loads(
            (self.output_dir / "_failed_paper_ids.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failed_ids, [])
        node = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(node["domain"]["domains"], ["synthetic"])

    def test_mock_endpoint_failure_returns_nonzero(self):
        with (
            mock.patch.object(
                extractor.requests,
                "post",
                side_effect=extractor.requests.ConnectionError(
                    "synthetic transport failure"
                ),
            ) as post,
            mock.patch("time.sleep"),
        ):
            exit_code = self._run_batch()

        self.assertEqual(post.call_count, 3)
        self.assertEqual(exit_code, 1)
        failed_ids = json.loads(
            (self.output_dir / "_failed_paper_ids.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failed_ids, ["case001"])
        self.assertFalse((self.output_dir / "case001_paper_node.json").exists())


if __name__ == "__main__":
    unittest.main()
