import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.dataset import load_jsonl


class EvaluationDatasetTests(unittest.TestCase):
    def write_dataset(self, records: list[object]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "dataset.jsonl"
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )
        return path

    def test_loads_and_normalizes_valid_record(self) -> None:
        path = self.write_dataset(
            [
                {
                    "question": "  Soru? ",
                    "answer": " Cevap ",
                    "contexts": [" Bağlam 1 ", "Bağlam 2"],
                    "ground_truth": " Referans ",
                }
            ]
        )

        records = load_jsonl(path)

        self.assertEqual(records[0].question, "Soru?")
        self.assertEqual(records[0].contexts, ("Bağlam 1", "Bağlam 2"))

    def test_rejects_missing_required_field(self) -> None:
        path = self.write_dataset(
            [{"question": "Soru?", "answer": "Cevap", "contexts": ["Bağlam"]}]
        )

        with self.assertRaisesRegex(ValueError, "ground_truth"):
            load_jsonl(path)

    def test_rejects_empty_contexts(self) -> None:
        path = self.write_dataset(
            [
                {
                    "question": "Soru?",
                    "answer": "Cevap",
                    "contexts": [],
                    "ground_truth": "Referans",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "contexts"):
            load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
