"""Convert SemEval .wa annotations and iSTS JSON from the command line."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .core import (
    ISTSFormatError,
    SentenceAnnotation,
    SentenceContext,
    context_from_input_files,
    read_json_records,
    serialize_wa_file,
    validate_prediction,
    wa_to_json,
    write_jsonl,
)


def _index_records(rows, name):
    """Index input records by ID and reject duplicates."""
    indexed = {}
    for row in rows:
        record_id = row.get("id")
        if type(record_id) not in (str, int) or not str(record_id).strip():
            raise ISTSFormatError(f"{name} record requires a nonempty id")
        key = str(record_id)
        if key in indexed:
            raise ISTSFormatError(f"duplicate {name} id: {key}")
        indexed[key] = row
    return indexed


def _contexts_from_rows(rows):
    """Load indexed sentence contexts from exported records."""
    return {
        key: SentenceContext.from_dict(row["context"])
        for key, row in _index_records(rows, "context").items()
    }


def _predictions_from_rows(rows, contexts):
    """Resolve batch prediction IDs or match a bare payload to one context."""
    if len(rows) == len(contexts) == 1 and set(rows[0]) == {"alignments"}:
        return {next(iter(contexts)): rows[0]}

    predictions = {}
    for key, row in _index_records(rows, "prediction").items():
        if "prediction" in row and "alignments" in row:
            raise ISTSFormatError(f"prediction {key} has both prediction and alignments fields")
        predictions[key] = row["prediction"] if "prediction" in row else {"alignments": row.get("alignments")}
    return predictions


def _annotations_from_json(contexts, predictions):
    """Join contexts to predictions and collect per-pair validation failures."""
    unexpected = sorted(predictions.keys() - contexts.keys())
    if unexpected:
        raise ISTSFormatError(f"predictions with unknown contexts: {unexpected[:5]}")
    annotations = []
    failures = []
    for key, context in contexts.items():
        try:
            if key not in predictions:
                raise ISTSFormatError("missing prediction")
            alignments = validate_prediction(predictions[key], context)
        except ISTSFormatError as exc:
            failures.append(f"{key}: {exc}")
            alignments = ()
        annotations.append(SentenceAnnotation(context, alignments))
    return annotations, failures


def _write_records(path, rows):
    """Write a JSON array or JSON Lines according to the output suffix."""
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        write_jsonl(path, rows)


def _from_wa(args):
    """Export .wa records as JSON, with optional single-pair prediction output."""
    records = wa_to_json(args.wa)
    if args.prediction_only:
        if len(records) != 1:
            raise ISTSFormatError("--prediction-only requires exactly one sentence pair")
        args.out.write_text(json.dumps(records[0]["prediction"], ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        _write_records(args.out, records)
    print(f"{len(records)} sentence pairs -> {args.out}")


def _contexts_from_raw(args):
    """Export sentence contexts from the two raw input files."""
    contexts = context_from_input_files(args.sent1, args.sent2)
    records = []
    for context in contexts:
        key = f"{args.id_prefix}:{context.sentence_id}" if args.id_prefix else context.sentence_id
        records.append({"id": key, "context": context.to_dict()})
    _write_records(args.out, records)
    print(f"{len(contexts)} contexts -> {args.out}")


def _to_wa(args):
    """Validate predictions and write .wa according to the invalid-reply policy."""
    contexts = _contexts_from_rows(read_json_records(args.contexts))
    predictions = _predictions_from_rows(read_json_records(args.predictions), contexts)
    annotations, failures = _annotations_from_json(contexts, predictions)
    if failures:
        print(f"Invalid/missing predictions: {len(failures)}/{len(contexts)}", file=sys.stderr)
        for message in failures[:20]:
            print(message, file=sys.stderr)
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more", file=sys.stderr)
        if args.invalid_policy == "error":
            raise ISTSFormatError("conversion rejected because of invalid predictions")
    serialize_wa_file(annotations, args.out)
    print(f"{len(annotations)} sentence pairs -> {args.out}, invalid/missing={len(failures)}")


def _run_perl(script, *files):
    """Run a Perl script on the given files and return its output."""
    result = subprocess.run(
        ["perl", str(script), "--", *map(str, files)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ISTSFormatError(f"{script.name}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def _compare_perl_scores(evaluator_dir, original, reconstructed):
    """Check well-formedness and compare reconstructed scores with the original."""
    validator = evaluator_dir / "wellformed.pl"
    print(_run_perl(validator, reconstructed))
    for name in ("evalF1_no_penalty.pl", "evalF1_penalty.pl"):
        script = evaluator_dir / name
        output = _run_perl(script, original, reconstructed)
        scores = re.findall(r"^\s*F1\s+\S+\s+(\d+\.\d+)", output, re.MULTILINE)
        if len(scores) != 4:
            raise ISTSFormatError(f"{name} did not return four F1 scores")
        if any(float(score) != 1 for score in scores):
            # A reference with zero links has self-F1 zero.
            baseline = _run_perl(script, original, original)
            if baseline != output:
                raise ISTSFormatError(f"{name} scored the reconstructed file differently from the original")
        print(f"{name}\n{output}")


def _roundtrip(args):
    """Verify that exporting to JSON and back preserves each .wa annotation."""
    if args.out and len(args.wa) != 1:
        raise ISTSFormatError("--out requires exactly one input .wa")

    for original in args.wa:
        records = wa_to_json(original)
        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "annotations.jsonl"
            reconstructed = Path(directory) / "roundtrip.wa"
            write_jsonl(json_path, records)

            exported = read_json_records(json_path)
            contexts = _contexts_from_rows(exported)
            predictions = _predictions_from_rows(exported, contexts)
            annotations, failures = _annotations_from_json(contexts, predictions)
            if failures:
                raise ISTSFormatError(f"round-trip JSON validation failed: {failures[0]}")
            serialize_wa_file(annotations, reconstructed)
            if wa_to_json(reconstructed) != records:
                raise ISTSFormatError(f"round-trip mismatch in {original}")
            if args.evaluator_dir:
                _compare_perl_scores(args.evaluator_dir, original, reconstructed)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reconstructed, args.out)
        print(f"{original.name}: {len(records)} sentence pairs survived .wa -> JSON -> .wa")


def main():
    """Parse converter commands and dispatch to the selected handler."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    from_wa = commands.add_parser("from-wa", help="Convert .wa to JSON with sentence contexts")
    from_wa.add_argument("wa", type=Path)
    from_wa.add_argument("-o", "--out", type=Path, required=True)
    from_wa.add_argument("--prediction-only", action="store_true", help="Export one bare prediction object")
    from_wa.set_defaults(handler=_from_wa)

    contexts = commands.add_parser("contexts-from-raw", help="Create contexts from raw sentence files")
    contexts.add_argument("--sent1", type=Path, required=True)
    contexts.add_argument("--sent2", type=Path, required=True)
    contexts.add_argument("-o", "--out", type=Path, required=True)
    contexts.add_argument("--id-prefix", default="")
    contexts.set_defaults(handler=_contexts_from_raw)

    to_wa = commands.add_parser("to-wa", help="Convert sentence contexts and JSON predictions to .wa")
    to_wa.add_argument("--contexts", type=Path, required=True)
    to_wa.add_argument("--predictions", type=Path, required=True)
    to_wa.add_argument("-o", "--out", type=Path, required=True)
    to_wa.add_argument(
        "--invalid-policy", choices=("error", "empty"), default="error",
        help="Reject invalid predictions or emit empty alignments for their pairs",
    )
    to_wa.set_defaults(handler=_to_wa)

    roundtrip = commands.add_parser("roundtrip", help="Check .wa -> JSON -> .wa")
    roundtrip.add_argument("wa", type=Path, nargs="+")
    roundtrip.add_argument("--evaluator-dir", type=Path, help="Directory containing the Perl evaluators")
    roundtrip.add_argument("--out", type=Path, help="Save the reconstructed .wa for one input file")
    roundtrip.set_defaults(handler=_roundtrip)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
