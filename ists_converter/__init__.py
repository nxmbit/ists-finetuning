"""Convert SemEval iSTS annotations between .wa and JSON."""

from .core import (
    Alignment,
    ISTSFormatError,
    SentenceAnnotation,
    SentenceContext,
    context_from_input_files,
    json_to_wa,
    parse_wa,
    parse_wa_file,
    read_json_records,
    serialize_wa,
    serialize_wa_file,
    validate_prediction,
    wa_to_json,
    write_jsonl,
)

__all__ = [
    "Alignment",
    "ISTSFormatError",
    "SentenceAnnotation",
    "SentenceContext",
    "context_from_input_files",
    "json_to_wa",
    "parse_wa",
    "parse_wa_file",
    "read_json_records",
    "serialize_wa",
    "serialize_wa_file",
    "validate_prediction",
    "wa_to_json",
    "write_jsonl",
]
