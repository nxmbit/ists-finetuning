import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ists_converter import ISTSFormatError, SentenceContext, json_to_wa, parse_wa, read_json_records


ROOT = Path(__file__).resolve().parents[1]


class ConverterTests(unittest.TestCase):
    def test_cli_bare_prediction_both_directions(self):
        context = SentenceContext.from_raw(7, "The dog ran", "The animal ran")
        prediction = {
            "alignments": [
                {"source_indices": [1, 2], "target_indices": [1], "label": "SPE1", "score": 4},
                {"source_indices": [3], "target_indices": [], "label": "NOALI_FACT", "score": 0},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_path = root / "context.json"
            prediction_path = root / "prediction.json"
            wa_path = root / "prediction.wa"
            restored_path = root / "restored.json"

            def convert(*args):
                subprocess.run([sys.executable, "-m", "ists_converter", *args], check=True, capture_output=True)

            context_path.write_text(json.dumps({"id": "example:7", "context": context.to_dict()}))
            prediction_path.write_text(json.dumps(prediction, indent=2))
            convert("to-wa", "--contexts", context_path, "--predictions", prediction_path, "-o", wa_path)
            self.assertIn("3 <==> 0 // NOALI_FACT // NIL", wa_path.read_text())
            convert("from-wa", wa_path, "--prediction-only", "-o", restored_path)
            self.assertEqual(json.loads(restored_path.read_text()), prediction)

            convert("from-wa", wa_path, "-o", restored_path)
            self.assertEqual(read_json_records(restored_path)[0]["context"], context.to_dict())

    def test_parser_rejects_duplicate_sections_and_sentence_ids(self):
        context = SentenceContext.from_raw(1, "source", "target")
        wa = json_to_wa(context, {"alignments": []})
        duplicate_section = wa.replace("</alignment>", "</alignment>\n<alignment>\n</alignment>")
        with self.assertRaisesRegex(ISTSFormatError, "alignment"):
            parse_wa(duplicate_section)
        with self.assertRaisesRegex(ISTSFormatError, "duplicate sentence ID"):
            parse_wa(wa + "\n" + wa)

    def test_empty_annotations_roundtrip_through_perl(self):
        context = SentenceContext.from_raw(1, "source", "target")
        with tempfile.TemporaryDirectory(prefix="iSTS converter ") as directory:
            path = Path(directory) / "empty.wa"
            path.write_text(json_to_wa(context, {"alignments": []}) + "\n")
            result = subprocess.run(
                [sys.executable, "-m", "ists_converter", "roundtrip", path, "--evaluator-dir", ROOT / "scripts"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.count("0.0000"), 8)
