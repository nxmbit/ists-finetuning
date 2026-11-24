#!/usr/bin/env python3
"""Evaluate a LiteLLM model on the SemEval 2016 iSTS test set.

The script uses the selected prompt to request replies for uncached sentence
pairs, converts them to .wa, and scores them with both Perl evaluators.
Use --workers N to send up to N requests concurrently for the active domain.

Environment variables
  OPENAI_API_KEY    LiteLLM proxy key if the proxy requires authentication
  OPENAI_BASE_URL   Proxy API base URL, e.g. http://localhost:4000/v1

--base-url overrides OPENAI_BASE_URL. The local default is http://localhost:4000/v1.

Outputs go to the printed run directory (default: benchmark_results/ists-*).
With --prompt-dir, each prompt has a subdirectory named after its file stem.
  results.csv                     F1 scores and failure counts per domain
  run.json, prompt.txt            Saved settings and prompt
  <domain>/raw/<id>.json          Cached API replies
  <domain>/predictions.wa         Predictions used for scoring
  <domain>/review.json            Failed pair IDs and reasons
  <domain>/malformed/<id>.txt     Reply text that failed conversion

Resume with the same options and --run-dir. Cached replies are reused and
the progress bar counts only calls for missing replies. To fix a reply, copy
its malformed file to <domain>/repairs/<id>.txt, edit it, and resume to rescore.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from openai import OpenAI, OpenAIError, omit
from openai.types.chat import ChatCompletion
from tqdm.auto import tqdm

from ists_converter import SentenceContext, json_to_wa
from ists_converter.jacyna import reply_to_wa


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/test_goldStandard"
SCRIPTS = ROOT / "scripts"
NO_PENALTY = SCRIPTS / "evalF1_no_penalty.pl"
PENALTY = SCRIPTS / "evalF1_penalty.pl"
VALIDATOR = SCRIPTS / "wellformed.pl"
DOMAINS = ("answers-students", "images", "headlines")
SHOTS = {"zero-shot": 0, "one-shot": 1, "eight-shot": 8}
CHUNK_MODES = ("system", "golden")
JSON_FORMATS = ("json", "jacyna-json")


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Preserve the run guide's layout and show defaults for command options."""


def arguments(argv):
    """Parse benchmark options and reject unsupported prompt or API settings."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--model", required=True, help="Model alias configured on the proxy")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or
                   "http://localhost:4000/v1", help="LiteLLM proxy API base, including any path prefix")
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", type=Path, help="Prompt file ending in .ists.txt, .json.txt or .jacyna-json.txt")
    prompts.add_argument("--prompt-dir", type=Path, help="Run every .txt prompt directly in this directory")
    p.add_argument("--chunks", choices=CHUNK_MODES, help="Input segmentation for prompts outside system/ or golden/")
    p.add_argument("--domain", choices=(*DOMAINS, "all"),
                   help="Default follows the report: all three for eight-shot, Answers otherwise")
    p.add_argument("--limit", type=int, help="Evaluate only the first N pairs per selected domain")
    p.add_argument("--workers", type=int, default=1, help="Maximum concurrent LLM requests")
    p.add_argument("--temperature", type=float, help="Sampling temperature, omitted from request by default")
    p.add_argument("--seed", type=int, help="Sampling seed for models that support it")
    p.add_argument("--max-tokens", type=int, help="Optional completion token limit")
    p.add_argument("--run-dir", type=Path, help="New output directory or an existing matching run to resume")
    args = p.parse_args(argv)
    if any(v is not None and v < 1 for v in (args.limit, args.max_tokens)):
        p.error("--limit and --max-tokens must be positive")
    if args.workers < 1:
        p.error("--workers must be positive")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        p.error("--temperature must be between 0 and 2")
    url = urlsplit(args.base_url)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        p.error("--base-url must be an HTTP(S) API base without credentials, query or fragment")
    return args


@dataclass(frozen=True)
class Prompt:
    """Store a prompt file and the benchmark settings inferred from its path."""

    path: Path
    chunks: str
    format: str
    shots: Optional[int]
    text: str
    sha256: str

    @classmethod
    def from_path(cls, path, chunks):
        """Read a prompt and determine its response format and input segmentation."""
        path = Path(os.path.abspath(path))
        if not path.is_file():
            raise ValueError(f"Prompt file not found: {path}")
        name, _, fmt = path.stem.rpartition(".")
        if path.suffix != ".txt" or fmt not in ("ists", *JSON_FORMATS):
            raise ValueError(f"{path.name}: use a .ists.txt, .json.txt or .jacyna-json.txt suffix")
        folder_mode = next((p.name for p in path.parents if p.name in CHUNK_MODES), None)
        if chunks and folder_mode and chunks != folder_mode:
            raise ValueError(f"{path.name}: --chunks {chunks} conflicts with its {folder_mode}/ folder")
        if not (chunks or folder_mode):
            raise ValueError(f"{path.name}: supply --chunks system or --chunks golden for this path")
        contents = path.read_bytes()
        text = contents.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Prompt file is empty: {path}")
        return cls(path, chunks or folder_mode, fmt, SHOTS.get(name), text, digest(contents))


@dataclass(frozen=True)
class Pair:
    """Hold a test pair's raw text, model input, and gold .wa record."""

    id: int
    raw: tuple[str, str]
    inputs: tuple[str, str]
    gold: str

    def context(self):
        """Build conversion context from the raw text and sentence ID."""
        return SentenceContext.from_raw(self.id, *self.raw)

    def message(self):
        """Format the user message with the selected input segmentation."""
        return (f"Sentence 1:\n{self.inputs[0]}\n\nSentence 2:\n{self.inputs[1]}\n\n"
                f"Now analyse these two sentences according to the iSTS instructions. Current sentence id is: {self.id}")


def load_pairs(domain, chunks, limit):
    """Load one test domain and check that its inputs match the gold record IDs."""
    prefix = f"STSint.testinput.{domain}"

    def inputs(suffix):
        """Read both sentence files for the requested chunk suffix."""
        return [(DATA / f"{prefix}.sent{side}{suffix}.txt").read_text(encoding="utf-8").splitlines()
                for side in (1, 2)]

    raw = inputs("")
    shown = inputs(".chunk") if chunks == "golden" else raw
    gold = (DATA / f"{prefix}.wa").read_text(encoding="utf-8")
    records = list(re.finditer(r'(?ms)^<sentence id="(\d+)"[^>]*>.*?</sentence>', gold))
    if not records or any(len(side) != len(records) for side in (*raw, *shown)):
        raise ValueError(f"{domain}: test inputs and reference IDs differ")
    pairs = []
    for i, (s1, s2, input1, input2, block) in enumerate(zip(*raw, *shown, records), start=1):
        if int(block[1]) != i or not all(s.strip() for s in (s1, s2, input1, input2)):
            raise ValueError(f"{domain} pair {i}: mismatched reference ID or blank input")
        pairs.append(Pair(i, (s1, s2), (input1.strip(), input2.strip()), block[0]))
    return pairs[:limit]


def completion(client, model, instructions, pair, request_options):
    """Request one chat completion and check that it contains a model message."""
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": pair.message()}],
        **request_options,
    )
    if not isinstance(reply, ChatCompletion):
        raise ValueError("Proxy returned an invalid chat completion envelope")
    payload = reply.to_dict()
    try:
        message = payload["choices"][0]["message"]
        if not isinstance(message, dict) or "content" not in message:
            raise KeyError("message.content")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Proxy returned an invalid chat completion envelope") from exc
    return payload


def prediction(pair, text, fmt):
    """Convert a model reply using the dataset pair's ID and raw token order.

    Validate canonical JSON replies directly. The Jacyna adapter handles
    archived JSON and iSTS replies. Return a newline-terminated .wa record.
    """
    context = pair.context()
    if fmt == "json":
        if not isinstance(text, str):
            raise ValueError("expected a text reply containing a JSON object")
        return json_to_wa(context, json.loads(text)) + "\n"
    return reply_to_wa(context, text, fmt) + "\n"


def write_text(path, text):
    """Atomically replace a run artifact with UTF-8 text."""
    # Each path has one writer. Publish only complete files.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def digest(data):
    """Return the SHA-256 fingerprint of a run input."""
    return hashlib.sha256(data).hexdigest()


def perl_scores(script, gold, predictions):
    """Run one Perl evaluator and return its four reported F1 values."""
    result = subprocess.run(
        ["perl", str(script), "--", str(gold), str(predictions)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"{script.name} failed: {result.stderr.strip()}")

    scores = dict(re.findall(
        r"^\s*F1 (Ali|Type|Score|Typ\+Sco)\s+(\d+\.\d+)\s*$",
        result.stdout,
        re.MULTILINE,
    ))
    if len(scores) != 4:
        raise RuntimeError(f"Incomplete output from {script.name}: {result.stdout}")
    return scores


def score(pairs, predictions):
    """Score predictions with both Perl evaluators against the selected pairs."""
    # Score only the selected gold records so --limit uses the same IDs on both sides.
    with tempfile.TemporaryDirectory() as temp_dir:
        gold = Path(temp_dir) / "gold.wa"
        gold.write_text("\n".join(pair.gold for pair in pairs) + "\n", encoding="utf-8")
        standard = perl_scores(NO_PENALTY, gold, predictions)
        relaxed = perl_scores(PENALTY, gold, predictions)

    return {
        "f1_ali": standard["Ali"],
        "f1_type": standard["Type"],
        "f1_score": standard["Score"],
        "f1_type_score_no_penalty": standard["Typ+Sco"],
        "f1_type_score_penalty": relaxed["Typ+Sco"],
        "penalty_status": "ok",
    }


def cache_replies(client, args, domain, pairs, instructions, request_options, raw, progress):
    """Fetch uncached replies with a bounded pool and save each successful reply."""
    missing = (pair for pair in pairs if not (raw / f"{pair.id}.json").exists())

    def fetch(pair):
        """Request and save one reply in the file owned by this pair."""
        try:
            response = completion(client, args.model, instructions, pair, request_options)
            write_text(raw / f"{pair.id}.json", json.dumps(response, ensure_ascii=False))
        except (OpenAIError, ValueError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"{domain} pair {pair.id}: {exc}") from exc

    workers = min(args.workers, len(pairs))
    if workers <= 1:
        for pair in missing:
            fetch(pair)
            progress.update(1)
        return

    first_batch = list(islice(missing, workers))
    if not first_batch:
        return

    first_error = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch, pair) for pair in first_batch}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                try:
                    future.result()
                except RuntimeError as exc:
                    if first_error is None:
                        first_error = exc
                else:
                    progress.update(1)

            # Let active requests finish after an error. Avoid starting new pairs.
            if first_error is None:
                pending.update(pool.submit(fetch, pair) for pair in islice(missing, workers - len(pending)))

    if first_error is not None:
        raise first_error


def evaluate_domain(client, args, domain, pairs, prompt, request_options, directory, progress):
    """Reuse or fetch replies, apply repairs, and score one domain."""
    domain_dir = directory / domain
    raw = domain_dir / "raw"
    repairs = domain_dir / "repairs"
    malformed = domain_dir / "malformed"
    raw.mkdir(parents=True, exist_ok=True)
    repairs.mkdir(exist_ok=True)
    cache_replies(client, args, domain, pairs, prompt.text.strip(), request_options, raw, progress)

    blocks = []
    issues = {}
    corrected = 0
    malformed_json = 0
    for pair in pairs:
        response_file = raw / f"{pair.id}.json"
        response = json.loads(response_file.read_text(encoding="utf-8"))
        repair = repairs / f"{pair.id}.txt"
        has_repair = repair.exists()
        text = (repair.read_text(encoding="utf-8")
                if has_repair else response["choices"][0]["message"]["content"])
        try:
            block = prediction(pair, text, prompt.format)
            corrected += int(has_repair)
        except ValueError as exc:
            issues[pair.id] = str(exc)
            malformed_json += isinstance(exc, json.JSONDecodeError)
            malformed.mkdir(exist_ok=True)
            failed_text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
            write_text(malformed / repair.name, failed_text)
            block = json_to_wa(pair.context(), {"alignments": []}) + "\n"
        blocks.append(block)
    predictions = domain_dir / "predictions.wa"
    write_text(predictions, "\n".join(blocks))
    write_text(domain_dir / "review.json", json.dumps(issues, ensure_ascii=False, indent=2))
    return {
        "model": args.model,
        "domain": domain,
        "chunks": prompt.chunks,
        "shots": prompt.shots,
        "format": prompt.format,
        "seed": args.seed,
        "pairs": len(pairs),
        "invalid": len(issues),
        "json_conversion_failed": len(issues) if prompt.format in JSON_FORMATS else None,
        "malformed_json": malformed_json if prompt.format in JSON_FORMATS else None,
        "repaired": corrected,
        **score(pairs, predictions),
    }


def save_run_config(directory, config):
    """Record the run configuration and enforce compatible inference settings on resume."""
    manifest = directory / "run.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        # Older manifests stored both the previous format name and prompt filename.
        old_format = {"json": "canonical-json", "jacyna-json": "json"}.get(config["format"])
        if old_format and previous["format"] == old_format:
            prompt_path = Path(config["prompt"])
            stem = prompt_path.stem.rsplit(".", 1)[0]
            old_prompt = prompt_path.with_name(f"{stem}.{old_format}.txt")
            if previous["prompt"] == str(old_prompt):
                previous.update(format=config["format"], prompt=config["prompt"])
        # Cached model replies depend on the inference settings and dataset.
        # Scoring is rerun from those replies with the current Perl scripts.
        previous["scorer_sha256"] = config["scorer_sha256"]
        if previous != config:
            raise ValueError("Run settings or data differ from run.json. Choose a new --run-dir.")
    elif any(path.name not in (".run.lock", "run.json.tmp") for path in directory.iterdir()):
        raise ValueError("Choose an empty --run-dir or resume an existing iSTS run")
    write_text(manifest, json.dumps(config, ensure_ascii=False, indent=2))


def lock_directory(stack, directory):
    """Hold a run directory lock until the surrounding exit stack closes."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = stack.enter_context((directory / ".run.lock").open("a"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError(f"Another evaluation is using {directory}") from exc


def run_prompt(client, args, prompt, directory, datasets, request_options, progress):
    """Evaluate one prompt and save its CSV after each completed domain."""
    write_text(directory / "prompt.txt", prompt.text)
    rows = []
    for domain, pairs in datasets.items():
        progress.set_postfix_str(domain)
        row = evaluate_domain(client, args, domain, pairs, prompt, request_options, directory, progress)
        rows.append(row)
        tmp = directory / "results.csv.tmp"
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(directory / "results.csv")
        summary = (
            f"{prompt.path.name} / {domain}: Ali {row['f1_ali']}, Type {row['f1_type']}, "
            f"Score {row['f1_score']}, Type+Score {row['f1_type_score_no_penalty']} (no penalty), "
            f"{row['f1_type_score_penalty']} (penalty)"
        )
        if prompt.format in JSON_FORMATS:
            summary += (
                f"\n  JSON conversion failures: {row['json_conversion_failed']}/{row['pairs']} "
                f"({row['malformed_json']} with invalid JSON syntax)"
            )
        if row["invalid"]:
            summary += f"\n  Failed replies: {directory / domain / 'malformed'}"
        tqdm.write(summary)


def run(args):
    """Run one prompt or a directory of prompts with a shared call progress bar."""
    if not shutil.which("perl"):
        raise ValueError("Perl is required to run the bundled evaluators")
    if args.prompt_dir:
        paths = sorted(path for path in args.prompt_dir.glob("*.txt") if path.is_file())
        if not paths:
            raise ValueError(f"No .txt prompts found in {args.prompt_dir}")
    else:
        paths = [args.prompt]
    prompts = [Prompt.from_path(path, args.chunks) for path in paths]

    sampling = {
        name: getattr(args, name)
        for name in ("temperature", "seed", "max_tokens")
        if getattr(args, name) is not None
    }
    scorer_hashes = {
        "no_penalty": digest(NO_PENALTY.read_bytes()),
        "penalty": digest(PENALTY.read_bytes()),
        "wellformed": digest(VALIDATOR.read_bytes()),
    }
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        parent = ROOT / "benchmark_results"
        parent.mkdir(exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="ists-", dir=parent))
    print(f"Run directory: {run_dir}", flush=True)

    with ExitStack() as stack:
        lock_directory(stack, run_dir)
        if args.prompt_dir and (run_dir / "run.json").exists():
            raise ValueError("Choose a separate --run-dir for the prompt batch")

        runs = []
        total_calls = 0
        for prompt in prompts:
            directory = run_dir
            if args.prompt_dir:
                directory = run_dir / prompt.path.stem
                lock_directory(stack, directory)
            selected = args.domain or ("all" if prompt.shots == 8 else "answers-students")
            domains = DOMAINS if selected == "all" else (selected,)
            datasets = {domain: load_pairs(domain, prompt.chunks, args.limit) for domain in domains}
            prompt_name = prompt.path
            if prompt_name.is_relative_to(ROOT):
                prompt_name = prompt_name.relative_to(ROOT)
            save_run_config(directory, {
                "model": args.model,
                "base_url": args.base_url.rstrip("/"),
                "chunks": prompt.chunks,
                "shots": prompt.shots,
                "format": prompt.format,
                "prompt": str(prompt_name),
                "prompt_sha256": prompt.sha256,
                "sampling": sampling,
                "limit": args.limit,
                "datasets": {
                    domain: digest(json.dumps([(p.id, p.raw, p.inputs, p.gold) for p in pairs]).encode())
                    for domain, pairs in datasets.items()
                },
                "scorer_sha256": scorer_hashes,
            })
            for domain, pairs in datasets.items():
                raw = directory / domain / "raw"
                total_calls += sum(not (raw / f"{pair.id}.json").exists() for pair in pairs)
            runs.append((prompt, directory, datasets))

        api_key = os.getenv("OPENAI_API_KEY")
        request_options = sampling.copy()
        if not api_key:
            # The SDK requires a key and an explicit opt-out for keyless proxies.
            request_options["extra_headers"] = {"Authorization": omit}
        client = stack.enter_context(
            OpenAI(base_url=args.base_url, api_key=api_key or "unused", timeout=900, max_retries=2)
        )
        if not total_calls:
            print("All replies are cached. Scoring saved replies.", flush=True)
        progress = stack.enter_context(tqdm(
            total=total_calls, unit="call", desc="LLM calls", dynamic_ncols=True, disable=total_calls == 0,
        ))
        for index, (prompt, directory, datasets) in enumerate(runs, 1):
            progress.set_description(f"[{index}/{len(runs)}] {prompt.path.name}")
            run_prompt(client, args, prompt, directory, datasets, request_options, progress)
    return run_dir


if __name__ == "__main__":
    try:
        run(arguments(sys.argv[1:]))
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(str(error))
