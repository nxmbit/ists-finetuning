"""Adapt replies to Jacyna's archived prompts for the shared iSTS converter."""

import json
import re

from .core import SentenceContext, json_to_wa


def _parse_indices(value, size):
    """Read a list or whitespace-separated span of model-generated token IDs.

    Expand a range such as `2-4` into `[2, 3, 4]` and map a lone `0` to
    an empty side. Reject duplicate IDs and ranges outside the sentence.
    """
    if not isinstance(value, (str, list)):
        raise ValueError("expected a list of token IDs")
    items = value.split() if isinstance(value, str) else value
    ids = []
    for item in items:
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", str(item))
        if not match:
            raise ValueError(f"invalid token ID {item!r}")
        first, last = int(match[1]), int(match[2] or match[1])
        if not 0 <= first <= last <= size:
            raise ValueError(f"token ID {item} outside 0..{size} or reversed")
        ids.extend(range(first, last + 1))
    if len(set(ids)) != len(ids) or (0 in ids and len(ids) != 1):
        raise ValueError("duplicate token IDs or mixed null and real IDs")
    return [] if ids == [0] else ids


def _normalize_alignment(source, target, label, score, context):
    """Build a canonical alignment row from an archived JSON or iSTS reply.

    Expand both token spans, uppercase the label, and normalize scores such
    as `4.0` to integers. For NOALI and ALIC, `NIL` or JSON null means zero.
    The shared converter checks the resulting row before writing .wa.
    """
    label = str(label).strip().upper()
    score = "NIL" if score is None else str(score).strip().upper()
    if score == "NIL" and label.split("_")[0] in ("NOALI", "ALIC"):
        score = "0"
    if not re.fullmatch(r"[0-5](?:\.0+)?", score):
        raise ValueError(f"invalid integer score {score} for {label}")
    return {
        "source_indices": _parse_indices(source, len(context.source_words)),
        "target_indices": _parse_indices(target, len(context.target_words)),
        "label": label,
        "score": int(score[0]),
    }


def _parse_json_reply(text, context):
    """Decode a JSON reply and recognize both supported alignment schemas.

    Translate Jacyna's `source_ids` and `translation_ids` into the canonical
    index fields. Pass canonical rows through for validation. The sentence
    context supplies the tokens when a reply includes its own token lists.
    """
    answer = json.loads(text)
    if not isinstance(answer, dict) or not isinstance(answer.get("alignments"), list):
        return answer

    # The archived prompts include token listings and use different names for the indices.
    legacy_keys = {"source_ids", "translation_ids"}
    rows = answer["alignments"]
    if not any(isinstance(row, dict) and legacy_keys & row.keys() for row in rows):
        # The archived prompts may also return an empty alignment array with token listings.
        if not rows and answer.keys() <= {"alignments", "source", "translation"}:
            return {"alignments": []}
        return answer

    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not (legacy_keys & row.keys()):
            normalized.append(row)
            continue
        if {"source_indices", "target_indices"} & row.keys():
            raise ValueError("alignment mixes Jacyna and converter index fields")
        try:
            normalized.append(_normalize_alignment(
                row["source_ids"], row["translation_ids"], row["label"], row["score"], context
            ))
        except KeyError as exc:
            raise ValueError(f"missing JSON field {exc}") from exc
    return {"alignments": normalized}


def _parse_ists_reply(text, context):
    """Read alignment rows from exactly one model-generated <alignment> block.

    The reply can omit the rest of the .wa record and the comment after each
    row. Ignore reply text outside the block and use the supplied context for
    token bounds. Normalize model shorthand such as `2-4` and `4.0`.
    """
    sections = re.findall(r"<alignment>(.*?)</alignment>", text, re.DOTALL | re.IGNORECASE)
    if len(sections) != 1 or len(re.findall(r"<alignment>", text, re.IGNORECASE)) != 1:
        raise ValueError("expected one complete <alignment> section")
    rows = []
    for line in sections[0].splitlines():
        if not line.strip():
            continue
        alignment, *fields = line.split("//")
        if len(fields) not in (2, 3) or alignment.count("<==>") != 1:
            raise ValueError(f"malformed alignment row: {line}")
        left, right = alignment.split("<==>")
        label, score = fields[:2]
        rows.append(_normalize_alignment(left, right, label, score, context))
    return {"alignments": rows}


def reply_to_wa(context: SentenceContext, text: str, fmt: str) -> str:
    """Convert one raw model reply into a complete .wa record.

    Accept the selected `jacyna-json` or `ists` format and unwrap a surrounding
    Markdown fence. Use the caller's sentence ID and tokens to build the
    output, then let the shared converter validate and render the alignments.
    Invalid JSON syntax raises JSONDecodeError so callers can count it separately
    from alignment validation failures.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty or non-text model answer")
    text = text.strip()
    fenced = re.fullmatch(r"```[^\n]*\n(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        text = fenced[1].strip()
    payload = _parse_json_reply(text, context) if fmt == "jacyna-json" else _parse_ists_reply(text, context)
    return json_to_wa(context, payload)
