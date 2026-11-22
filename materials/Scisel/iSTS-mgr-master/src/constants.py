import os
import pathlib

ROOT_DIR = os.getcwd()
DATA_PATH = pathlib.Path(ROOT_DIR + "/Data/")
TRAIN_DATA_PATH = DATA_PATH / "train_data"
TEST_DATA_PATH = DATA_PATH / "test_data"

RAW_DATA = DATA_PATH / 'raw'
CHUNKED_DATA  = DATA_PATH / 'chunked'
WA_DATA = DATA_PATH / 'wa'

DATASETS = ["answers-students", "headlines", "images"]

INSTRUCTIONS = {
    "alignment":"Your task is to label the alignment of pair of given chunks, " \
                "based on their semantic relation to each other. Return only ONE of the possible labels, nothing more. " \
                "Here are the possible labels:\n",

    "score": "Your task is to evaluate and score the alignment" \
             "of two provided chunks of text." \
             "The goal is to assess their degree of similarity" \
             " in meaning, structure, or relevance to each other. Answer format: LABEL=<one_label> only. Output exactly ONE value from following scale:",
    "merged": (
    "Your task is to jointly predict:\n"
    "(1) a SCORE for how well two text chunks align,\n"
    "(2) an ALIGNMENT label describing their semantic relation.\n\n"
    "Return ONLY a single output in this exact format:\n"
    "SCORE-ALIGNMENT\n\n"
    "Rules:\n"
    "- Output exactly one line.\n"
    "- No extra text, no explanation, no punctuation besides the single hyphen.\n"
    "- No spaces.\n"
    "- Return exactly one pair.\n"
    "- If you are uncertain, still choose the closest valid SCORE and ALIGNMENT; never output anything else.\n\n"
    " Possible outputs:\n"
    ),
    "both": "Your task is to label the alignment of pair of given chunks, " \
                "based on their semantic relation to each other also evaluating and scoring the alignment of two provided chunks of text" \
                "The goal is to assess their degree of similarity" \
             " in meaning, structure, or relevance to each other." \
             "Answer format: ALIGNMENT:=<one_label>; SCORE:=<one_label> only. Output exactly ONE value from each of following scales:"
}

LABELS_DESCRIPTION = {
    "alignment" :  {"detailed": """
EQUI: both chunks have the same meaning,they are semantically equivalent in this context.
OPPO: the meanings of the chunks are in opposition to each other, lying in an inherently incompatible binary relationship.
SPE1: both chunks have similar meanings, but chunk in sentence 1 is more specific.
SPE2: like SPE1, but it is the chunk in sentence 2 which is more specific.
SIMI: both chunks have similar meanings, theyshare similar attributes and there is no EQUI, OPPO, SPE1 or SPE2 relation.
REL: both chunks are not considered similar but they are closely related by some relation not mentioned above (i.e. no EQUI, OPPO, SPE1, SPE2, or SIMI relation).
NOALI: this chunk has not any corresponding chunk in the other sentence. Therefore, it is left unaligned.
                                    """,
    "simple": "[EQUI, OPPO, SPE1, SPE2, SIMI, REL, NOALI]"
    },
    "score": {
        "detailed": """
        0: No Relation.
        1: Slightly Related.
        2: Related but Different.
        3: Some Similarity.
        4: Strong Similarity.
        5: Completely Equivalent.
        """,
        "simple": "[0, 1, 2, 3, 4, 5]"
        },
    "merged": {
        "detailed": ("0-NOALI: No Relation / Unaligned. One chunk has no corresponding chunk in the other sentence. "
    "They do not match, do not describe the same idea, and there is no meaningful semantic link.\n\n"

    "5-EQUI: Completely Equivalent / Semantically Identical. Both chunks express the same meaning in this context. "
    "They are mutually substitutable without changing the intended interpretation.\n\n"

    "1-OPPO: Very weak opposition. Some tension or mismatch in stance, but not a clean contradiction; limited overlap.\n"
    "2-OPPO: Mild opposition. Clearly incompatible implications on a related topic, but scope/conditions differ.\n"
    "3-OPPO: Moderate opposition. Mostly about the same claim/situation, but assert incompatible outcomes/values.\n"
    "4-OPPO: Strong opposition. Near-binary contradiction; highly overlapping content with reversed polarity.\n"
    "5-OPPO: Maximal opposition. Direct contradiction of the same proposition (as close to 'P' vs 'not P' as possible in context).\n\n"

    "1-SPE1: Very weak specificity (S1 more specific). Chunk 1 is only slightly narrower; overlap is limited; entailment is uncertain.\n"
    "2-SPE1: Mild specificity (S1 more specific). Chunk 1 adds some constraints/details, but notable differences remain.\n"
    "3-SPE1: Moderate specificity (S1 more specific). Chunk 1 mostly matches chunk 2 and adds meaningful constraints; generally entails chunk 2.\n"
    "4-SPE1: Strong specificity (S1 more specific). High overlap; chunk 1 is essentially chunk 2 plus qualifiers/narrower scope.\n"
    "5-SPE1: Maximal specificity (S1 more specific). Chunk 1 clearly and fully entails chunk 2; difference is almost only added precision (not equivalent).\n\n"

    "1-SPE2: Very weak specificity (S2 more specific). Chunk 2 is only slightly narrower; overlap is limited; entailment is uncertain.\n"
    "2-SPE2: Mild specificity (S2 more specific). Chunk 2 adds some constraints/details, but notable differences remain.\n"
    "3-SPE2: Moderate specificity (S2 more specific). Chunk 2 mostly matches chunk 1 and adds meaningful constraints; generally entails chunk 1.\n"
    "4-SPE2: Strong specificity (S2 more specific). High overlap; chunk 2 is essentially chunk 1 plus qualifiers/narrower scope.\n"
    "5-SPE2: Maximal specificity (S2 more specific). Chunk 2 clearly and fully entails chunk 1; difference is almost only added precision (not equivalent).\n\n"

    "1-SIMI: Very slight similarity. Small shared theme/attributes, but core meanings differ; far from substitutable.\n"
    "2-SIMI: Mild similarity. Clear topical/semantic overlap, yet important differences dominate.\n"
    "3-SIMI: Moderate similarity. Meaningfully similar intent/concept, but neither is a specific case of the other; not equivalent.\n"
    "4-SIMI: Strong similarity. Very close meanings, but a key difference prevents EQUI or specificity.\n"
    "5-SIMI: Maximal similarity (non-EQUI). Near-paraphrase with extremely high overlap, yet still not fully equivalent (a subtle but real semantic difference remains).\n\n"

    "1-REL: Very weak relation (not similar). Same broad domain/frame, but link is tenuous.\n"
    "2-REL: Mild relation (not similar). Identifiable connection (e.g., same situation or associated concepts), but meanings are different.\n"
    "3-REL: Moderate relation (not similar). Clearly connected by a specific relation (cause/effect, part/whole, question/answer, entity/attribute), but not similar.\n"
    "4-REL: Strong relation (not similar). Tightly linked within the same scenario; strongly relevant to each other, yet non-substitutable.\n"
    "5-REL: Maximal relation (not similar). Very tight, structurally meaningful linkage (e.g., claim/evidence, instruction/required action, definition/instance), but still not meaning-similar.\n"
    ),
    "simple": ("["
    "0-NOALI,"
    "5-EQUI,"
    "1-OPPO,"
    "2-OPPO,"
    "3-OPPO,"
    "4-OPPO,"
    "5-OPPO,"
    "1-SPE1,"
    "2-SPE1,"
    "3-SPE1,"
    "4-SPE1,"
    "5-SPE1,"
    "1-SPE2,"
    "2-SPE2,"
    "3-SPE2,"
    "4-SPE2,"
    "5-SPE2,"
    "1-SIMI,"
    "2-SIMI,"
    "3-SIMI,"
    "4-SIMI,"
    "5-SIMI,"
    "1-REL,"
    "2-REL,"
    "3-REL,"
    "4-REL,"
    "5-REL"
    "]"
)
    }
}
LABELS_DESCRIPTION["both"] = {
    "detailed": f"Alignment labels:\n{LABELS_DESCRIPTION["alignment"]["detailed"]}\nScore labels:\n{LABELS_DESCRIPTION["score"]["detailed"]}",
    "simple": f"Alignment labels:\n{LABELS_DESCRIPTION["alignment"]["simple"]}\nScore labels:\n{LABELS_DESCRIPTION["score"]["simple"]}"}

TASKS_LABELS = {"alignment": ["EQUI", "OPPO", "SPE1", "SPE2", "SIMI", "REL", "NOALI"],
               "score": ["0", "1", "2", "3", "4", "5"],
               "merged": [
    "0-NOALI",
    "5-EQUI",
    "1-OPPO", "2-OPPO", "3-OPPO", "4-OPPO", "5-OPPO",
    "1-SPE1", "2-SPE1", "3-SPE1", "4-SPE1", "5-SPE1",
    "1-SPE2", "2-SPE2", "3-SPE2", "4-SPE2", "5-SPE2",
    "1-SIMI", "2-SIMI", "3-SIMI", "4-SIMI", "5-SIMI",
    "1-REL",  "2-REL",  "3-REL",  "4-REL",  "5-REL",
]}

VLLM_MODEL = "openai/gpt-oss:120b"
VLLM_SYSTEM_PROMPT = "You are helpful assistant."

