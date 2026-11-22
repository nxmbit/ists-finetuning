from transformers import AutoModelForCausalLM
import torch
from src.constants import TASKS_LABELS
import re
from typing import Optional


def load_model(model_name, cache_dir, quant=None):

    if quant is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    elif quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    elif quant == "8bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, load_in_8bit=True)

    return model


def select_label_from_response(response: str, task_type: str) -> Optional[str]:
    """
    Extracts the predicted label from a model response.

    task_type:
      - "alignment" / "score": returns a single label found in TASKS_LABELS[task_type]
      - "both": returns a merged label in format "SCORE-ALIGNMENT" (e.g., "3-OPPO")

    Notes:
      - For "both", we first try to directly match a valid "SCORE-ALIGNMENT" pattern.
      - If that fails, we fall back to extracting SCORE and ALIGNMENT separately (robust to formats like
        "ALIGNMENT: OPPO; SCORE: 3" or "SCORE=3 ALIGNMENT=OPPO"), then merge.
    """
    if response is None:
        return None

    text = response.strip()

    if task_type != "both":
        possible_labels = TASKS_LABELS[task_type]
        occurrences = {}

        for possible_word in possible_labels:
            idx = text.find(possible_word)
            if idx != -1:
                occurrences[possible_word] = idx

        return min(occurrences, key=occurrences.get) if occurrences else None

    # ---- task_type == "both" ----
    possible_alignments = TASKS_LABELS["alignment"]
    possible_scores = TASKS_LABELS["score"]

    # 1) Best case: model already returned "SCORE-ALIGNMENT"
    #    - allow optional spaces around hyphen, but we output canonical "SCORE-ALIGNMENT"
    direct = re.search(r"\b([0-5])\s*-\s*([A-Z]{3,5}\d?)\b", text)
    if direct:
        score, alignment = direct.group(1), direct.group(2)

        if score in possible_scores and alignment in possible_alignments:
            return f"{score}-{alignment}"

    # 2) Fallback: extract alignment and score separately from messy outputs
    #    (e.g., "ALIGNMENT: OPPO; SCORE: 3" / "SCORE=3; LABEL=OPPO" etc.)
    score = None
    alignment = None

    # Find score (prefer explicit "score" mention)
    m_score = re.search(r"(?i)\bscore\b\s*[:=]?\s*([0-5])\b", text)
    if m_score and m_score.group(1) in possible_scores:
        score = m_score.group(1)
    else:
        # any standalone digit 0-5 (last resort)
        m_any = re.search(r"\b([0-5])\b", text)
        if m_any and m_any.group(1) in possible_scores:
            score = m_any.group(1)

    # Find alignment (prefer explicit "alignment" mention)
    m_align = re.search(r"(?i)\balignment\b\s*[:=]?\s*([A-Z]{3,5}\d?)\b", text)
    if m_align and m_align.group(1).upper() in possible_alignments:
        alignment = m_align.group(1).upper()
    else:
        # search by first occurrence among known alignment labels
        occurrences = {}
        for lab in possible_alignments:
            idx = text.find(lab)
            if idx != -1:
                occurrences[lab] = idx
        if occurrences:
            alignment = min(occurrences, key=occurrences.get)

    if score and alignment:
        merged = f"{score}-{alignment}"

        # Optional: enforce your constraint if you still want it
        # (comment out if all 0-5 with all labels are now allowed)
        # if merged != "0-NOALI" and score == "0":
        #     return None
        # if alignment == "NOALI" and score != "0":
        #     return None

        return merged

    return None