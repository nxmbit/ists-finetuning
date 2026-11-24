"""Convert SemEval iSTS annotations between .wa and JSON.

The JSON payload contains alignments. Sentence context supplies the ID and
tokens needed to write the XML-like .wa record.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class ISTSFormatError(ValueError):
    """Invalid .wa input, prediction schema, or index mapping."""


MAIN_LABELS = frozenset({"EQUI", "OPPO", "SPE1", "SPE2", "SIMI", "REL", "NOALI", "ALIC"})
OPTIONAL_LABELS = frozenset({"FACT", "POL"})
SENTENCE_RE = re.compile(r"<sentence\b([^>]*)>(.*?)</sentence>", re.DOTALL)
ATTRIBUTE_RE = re.compile(r'([a-zA-Z_][\w-]*)\s*=\s*"([^"]*)"')
WORD_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*:\s*$")
SENTENCE_BODY_RE = re.compile(
    r"\s*//[ \t]?([^\n]*)\n[ \t]*//[ \t]?([^\n]*)\n"
    r"\s*<source>(.*?)</source>"
    r"\s*<translation>(.*?)</translation>"
    r"\s*<alignment>(.*?)</alignment>\s*",
    re.DOTALL,
)
REQUIRED_ALIGNMENT_FIELDS = frozenset({"source_indices", "target_indices", "label", "score"})


def validate_label(label: str) -> None:
    """Require one base relation and distinct optional FACT and POL modifiers."""
    if not isinstance(label, str) or not label:
        raise ISTSFormatError("label must be a nonempty string")
    parts = label.split("_")
    if len(parts) > 3 or parts[0] not in MAIN_LABELS or any(x not in OPTIONAL_LABELS for x in parts[1:]):
        raise ISTSFormatError(f"invalid SemEval alignment label: {label!r}")
    if len(parts) != len(set(parts)):
        raise ISTSFormatError(f"duplicate alignment label component: {label!r}")


def _validate_indices(values: Any, *, length: int, name: str) -> tuple[int, ...]:
    """Check that a JSON array contains unique, in-range, 1-based token indices."""
    if not isinstance(values, list):
        raise ISTSFormatError(f"{name} must be a JSON array")
    result: list[int] = []
    for idx in values:
        if type(idx) is not int or not 1 <= idx <= length:
            raise ISTSFormatError(f"{name}: index {idx!r} is not an integer in 1..{length}")
        result.append(idx)
    if len(set(result)) != len(result):
        raise ISTSFormatError(f"{name}: repeated word index within one chunk")
    return tuple(result)


@dataclass(frozen=True)
class SentenceContext:
    """Store the ID, token order, and sentence text needed to write a .wa record."""

    sentence_id: str
    source_words: tuple[str, ...]
    target_words: tuple[str, ...]
    source_sentence: str
    target_sentence: str
    status: str = ""

    def __post_init__(self) -> None:
        """Check that the sentence metadata and indexed tokens agree."""
        if not isinstance(self.sentence_id, str) or not self.sentence_id.strip():
            raise ISTSFormatError("sentence_id must be a nonempty string")
        if not isinstance(self.status, str):
            raise ISTSFormatError("status must be a string")
        if any(char in value for value in (self.sentence_id, self.status) for char in '"<>\r\n'):
            raise ISTSFormatError("sentence ID and status cannot contain quotes, angle brackets or newlines")
        if not self.source_words or not self.target_words:
            raise ISTSFormatError("source and target must both contain words")
        if any(not isinstance(word, str) or not word for word in (*self.source_words, *self.target_words)):
            raise ISTSFormatError("word lists must contain nonempty strings")
        for sentence in (self.source_sentence, self.target_sentence):
            if not isinstance(sentence, str) or "\n" in sentence or "\r" in sentence:
                raise ISTSFormatError("sentence comments must be strings on a single line")
        # The evaluator reads words from these comment lines, splitting on spaces.
        if tuple(self.source_sentence.split()) != self.source_words:
            raise ISTSFormatError("source sentence comments and indexed source words disagree")
        if tuple(self.target_sentence.split()) != self.target_words:
            raise ISTSFormatError("target sentence comments and indexed target words disagree")

    @classmethod
    def from_raw(cls, sentence_id: str | int, first: str, second: str) -> "SentenceContext":
        """Use ONLY input sentences, never gold chunking/alignment (System Chunks)."""
        return cls(
            sentence_id=str(sentence_id),
            source_words=tuple(first.split()),
            target_words=tuple(second.split()),
            source_sentence=" ".join(first.split()),
            target_sentence=" ".join(second.split()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return sentence context as a JSON-serializable dictionary."""
        return {
            "sentence_id": self.sentence_id,
            "status": self.status,
            "source_sentence": self.source_sentence,
            "target_sentence": self.target_sentence,
            "source_words": list(self.source_words),
            "target_words": list(self.target_words),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "SentenceContext":
        """Restore and validate sentence context from a JSON object."""
        if not isinstance(obj, Mapping):
            raise ISTSFormatError("context must be a JSON object")
        try:
            sentence_id = obj["sentence_id"]
            source_words = obj["source_words"]
            target_words = obj["target_words"]
            if type(sentence_id) not in (str, int):
                raise ISTSFormatError("sentence_id must be a string or integer")
            if not isinstance(source_words, list) or not isinstance(target_words, list):
                raise ISTSFormatError("source_words and target_words must be JSON arrays")
            return cls(
                sentence_id=str(sentence_id),
                source_words=tuple(source_words),
                target_words=tuple(target_words),
                source_sentence=obj["source_sentence"],
                target_sentence=obj["target_sentence"],
                status=obj.get("status", ""),
            )
        except KeyError as exc:
            raise ISTSFormatError(f"missing sentence context field {exc}") from exc


@dataclass(frozen=True)
class Alignment:
    """Represent one scored relation between source and target token spans."""

    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    label: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        """Return one alignment in the canonical JSON schema."""
        return {
            "source_indices": list(self.source_indices),
            "target_indices": list(self.target_indices),
            "label": self.label,
            "score": self.score,
        }


@dataclass(frozen=True)
class SentenceAnnotation:
    """Group a sentence pair's context with its alignments."""

    context: SentenceContext
    alignments: tuple[Alignment, ...]

    def prediction_dict(self) -> dict[str, Any]:
        """Return the bare JSON prediction payload for this sentence pair."""
        return {"alignments": [alignment.to_dict() for alignment in self.alignments]}


def _validate_alignment(source, target, label, score, context: SentenceContext) -> Alignment:
    """Validate the token spans, relation, and score for one alignment."""
    source = _validate_indices(source, length=len(context.source_words), name="source_indices")
    target = _validate_indices(target, length=len(context.target_words), name="target_indices")
    validate_label(label)
    if type(score) is not int or not 0 <= score <= 5:
        raise ISTSFormatError(f"score must be an integer from 0 to 5, got {score!r}")
    if not source and not target:
        raise ISTSFormatError("an alignment cannot have both sides empty")

    main_label = label.split("_")[0]
    if source and target and main_label == "NOALI":
        raise ISTSFormatError("NOALI must have exactly one empty side")
    if (not source or not target) and main_label not in {"NOALI", "ALIC"}:
        raise ISTSFormatError("only NOALI or ALIC may have an empty side")
    return Alignment(source, target, label, score)


def validate_prediction(prediction: Any, context: SentenceContext) -> tuple[Alignment, ...]:
    """Validate the canonical JSON payload against the indexed sentence pair."""
    if not isinstance(prediction, Mapping) or set(prediction) != {"alignments"}:
        raise ISTSFormatError('prediction must be an object with exactly one key: "alignments"')
    rows = prediction["alignments"]
    if not isinstance(rows, list):
        raise ISTSFormatError("alignments must be a JSON array")
    alignments: list[Alignment] = []
    for row_number, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict) or set(row) != REQUIRED_ALIGNMENT_FIELDS:
                raise ISTSFormatError(f"expected exactly these fields: {sorted(REQUIRED_ALIGNMENT_FIELDS)}")
            alignments.append(_validate_alignment(
                row["source_indices"], row["target_indices"], row["label"], row["score"], context
            ))
        except ISTSFormatError as exc:
            raise ISTSFormatError(f"alignment #{row_number}: {exc}") from exc
    # Intentionally do not require complete word coverage: 3 records in original
    # SemEval test gold lack it. Many-to-many/overlapping alignments are allowed.
    return tuple(alignments)


def _parse_words(section_text: str, section_name: str) -> tuple[str, ...]:
    """Read a .wa token section and require consecutive indices starting at one."""
    words: list[str] = []
    for line in section_text.splitlines():
        if not line.strip():
            continue
        match = WORD_RE.fullmatch(line)
        if not match:
            raise ISTSFormatError(f"invalid word line in <{section_name}>: {line!r}")
        index, word = match.groups()
        if int(index) != len(words) + 1:
            raise ISTSFormatError(f"non-consecutive <{section_name}> word index: {index}")
        words.append(word)
    return tuple(words)


def _parse_wa_alignment(line: str, context: SentenceContext) -> Alignment:
    """Parse one alignment row from a complete SemEval .wa record.

    The row must contain both token spans, a label, a score, and a fourth
    field for the comment. Spans list individual token IDs. A lone `0`
    marks an unaligned side, and `NIL` maps to zero for NOALI and ALIC.
    The comment text is ignored. Indices are checked against the record's
    own token lists.
    """
    parts = line.split("//", 3)
    if len(parts) != 4:
        raise ISTSFormatError(f"expected four //-separated alignment fields: {line!r}")
    index_expr, label, score_string = (field.strip() for field in parts[:3])
    pair = index_expr.split("<==>")
    if len(pair) != 2:
        raise ISTSFormatError(f"invalid <==> alignment: {line!r}")

    def parse_side(side: str) -> list[int]:
        """Read a whitespace-separated token span or a single unaligned zero."""
        if not re.fullmatch(r"[0-9]+(?:\s+[0-9]+)*", side.strip()):
            raise ISTSFormatError(f"expected numeric indices or 0, got {side.strip()!r}")
        indices = [int(index) for index in side.split()]
        if indices == [0]:
            return []
        return indices

    source, target = parse_side(pair[0]), parse_side(pair[1])
    if score_string == "NIL":
        if label.split('_')[0] not in {"NOALI", "ALIC"}:
            raise ISTSFormatError("NIL score allowed only for NOALI/ALIC")
        score = 0
    else:
        try:
            score = int(score_string)
        except ValueError as exc:
            raise ISTSFormatError(f"score must be 0..5 or NIL, got {score_string!r}") from exc
        if str(score) != score_string:
            raise ISTSFormatError(f"score must be canonical integer 0..5, got {score_string!r}")
    return _validate_alignment(source, target, label, score, context)


def parse_wa(text: str) -> list[SentenceAnnotation]:
    """Read one or more complete SemEval .wa records.

    Each record supplies its own sentence ID, sentence comments, indexed
    source and translation tokens, and alignment block. This parser checks
    the record structure and returns the context and validated alignments.
    Use jacyna.reply_to_wa for raw model replies with an externally supplied
    sentence context.
    """
    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    annotations: list[SentenceAnnotation] = []
    matches = list(SENTENCE_RE.finditer(text))
    if not matches:
        raise ISTSFormatError("no <sentence> records found")
    residue = SENTENCE_RE.sub("", text)
    if residue.strip():
        raise ISTSFormatError(f"unexpected text outside <sentence> records: {residue[:100]!r}")
    seen_ids = set()
    for match in matches:
        attributes = ATTRIBUTE_RE.findall(match[1])
        attrs = dict(attributes)
        if len(attrs) != len(attributes):
            raise ISTSFormatError("duplicate <sentence> attribute")
        sentence_id = attrs.get("id")
        if not sentence_id:
            raise ISTSFormatError("<sentence> without id attribute")
        if sentence_id in seen_ids:
            raise ISTSFormatError(f"duplicate sentence ID {sentence_id}")
        seen_ids.add(sentence_id)
        try:
            body = SENTENCE_BODY_RE.fullmatch(match[2])
            if not body:
                raise ISTSFormatError("expected two sentence comments followed by source, translation and alignment")
            source_sentence, target_sentence, source, target, alignments = body.groups()
            context = SentenceContext(
                sentence_id, _parse_words(source, "source"), _parse_words(target, "translation"),
                source_sentence, target_sentence, attrs.get("status", ""),
            )
            rows = tuple(_parse_wa_alignment(line, context) for line in alignments.splitlines() if line.strip())
        except ISTSFormatError as exc:
            raise ISTSFormatError(f"sentence {sentence_id}: {exc}") from exc
        annotations.append(SentenceAnnotation(context, rows))
    return annotations


def parse_wa_file(path: str | Path) -> list[SentenceAnnotation]:
    """Load and parse all sentence annotations from a UTF-8 .wa file."""
    return parse_wa(Path(path).read_text(encoding="utf-8-sig"))


def _alignment_line(alignment: Alignment, context: SentenceContext) -> str:
    """Format one alignment row and reconstruct its token-span comments."""
    source = " ".join(map(str, alignment.source_indices)) or "0"
    target = " ".join(map(str, alignment.target_indices)) or "0"
    source_text = " ".join(context.source_words[i - 1] for i in alignment.source_indices) or "-not aligned-"
    target_text = " ".join(context.target_words[i - 1] for i in alignment.target_indices) or "-not aligned-"
    score = str(alignment.score)
    if alignment.score == 0 and (source == "0" or target == "0"):
        score = "NIL"
    return f"{source} <==> {target} // {alignment.label} // {score} // {source_text} <==> {target_text}"


def serialize_wa(annotation: SentenceAnnotation) -> str:
    """Format one sentence annotation as a validated .wa record."""
    return json_to_wa(annotation.context, annotation.prediction_dict())


def json_to_wa(context: SentenceContext, prediction: Mapping[str, Any]) -> str:
    """Validate a canonical JSON prediction and render one .wa sentence record."""
    rows = validate_prediction(prediction, context)
    lines = [
        f'<sentence id="{context.sentence_id}" status="{context.status}">',
        f"// {context.source_sentence}",
        f"// {context.target_sentence}",
    ]
    for tag, words in (("source", context.source_words), ("translation", context.target_words)):
        lines.append(f"<{tag}>")
        lines.extend(f"{i} {word} :" for i, word in enumerate(words, 1))
        lines.append(f"</{tag}>")

    lines.append("<alignment>")
    lines.extend(_alignment_line(row, context) for row in rows)
    lines.extend(["</alignment>", "</sentence>"])
    return "\n".join(lines)


def serialize_wa_file(annotations: Iterable[SentenceAnnotation], path: str | Path) -> None:
    """Write a nonempty collection of annotations with unique sentence IDs."""
    rows = list(annotations)
    if not rows:
        raise ISTSFormatError('cannot serialize an empty annotation collection')
    ids = [a.context.sentence_id for a in rows]
    if len(ids) != len(set(ids)):
        raise ISTSFormatError('duplicate sentence IDs in output .wa (separate the domains)')
    Path(path).write_text("\n\n".join(serialize_wa(annotation) for annotation in rows) + "\n", encoding="utf-8")


def context_from_input_files(sent1: str | Path, sent2: str | Path) -> list[SentenceContext]:
    """Build contexts from matching lines in the two raw sentence files."""
    first = Path(sent1).read_text(encoding='utf-8-sig').splitlines()
    second = Path(sent2).read_text(encoding='utf-8-sig').splitlines()
    if len(first) != len(second):
        raise ISTSFormatError(f"sentence input count mismatch: {len(first)} vs {len(second)}")
    return [SentenceContext.from_raw(i, a, b) for i, (a, b) in enumerate(zip(first, second), 1)]


def wa_to_json(wa: str | Path) -> list[dict[str, Any]]:
    """Return the prediction payload alongside the context needed to restore it."""
    return [
        {"id": annotation.context.sentence_id,
         "context": annotation.context.to_dict(),
         "prediction": annotation.prediction_dict()}
        for annotation in parse_wa_file(wa)
    ]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write one UTF-8 JSON object per line."""
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects from an object, array, or JSON Lines file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    rows = None
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # Older converter versions could write JSONL to a .json file.
            if exc.msg != "Extra data":
                raise ISTSFormatError(f"{path}:{exc.lineno}: invalid JSON: {exc.msg}") from exc
        else:
            rows = payload if isinstance(payload, list) else [payload]
    if rows is None:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ISTSFormatError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ISTSFormatError(f"{path}: JSON records must be objects")
    return rows
