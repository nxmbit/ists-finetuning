import csv
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

import evaluate_ists as evaluator
from ists_converter import json_to_wa, parse_wa


def completion(text):
    body = {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
    return httpx.Response(200, json=body)


def proxy(handler):
    def client(**settings):
        return OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **settings)
    return patch.object(evaluator, "OpenAI", side_effect=client)


class EvaluationTests(unittest.TestCase):
    def test_reference_round_trip(self):
        # Includes an empty gold alignment, noncontiguous spans, M:N and EQUI_POL.
        for domain in evaluator.DOMAINS:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                pairs = evaluator.load_pairs(domain, "system", None)
                wa = Path(tmp) / "system.wa"
                wa.write_text("\n".join(evaluator.prediction(p, p.gold, "ists") for p in pairs))
                scores = evaluator.score(pairs, wa)
                self.assertEqual([v for k, v in scores.items() if k.startswith("f1_")], ["1.0000"] * 5)

    def test_chat_request_resume_and_repair(self):
        pairs = evaluator.load_pairs("answers-students", "golden", 2)
        with tempfile.TemporaryDirectory(prefix="iSTS run ") as tmp, redirect_stdout(io.StringIO()):
            args = evaluator.arguments([
                "--model", "proxy-alias", "--base-url", "http://localhost:4000/v1",
                "--prompt", "prompts/golden/one-shot.ists.txt", "--limit", "2", "--run-dir", tmp,
            ])
            refused = httpx.Response(401, json={"error": {"message": "bad key"}})
            transport = Mock(side_effect=[completion(pairs[0].gold), refused])
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), proxy(transport):
                with self.assertRaisesRegex(RuntimeError, "answers-students pair 2"):
                    evaluator.run(args)
                request = transport.call_args_list[0].args[0]
                self.assertEqual(str(request.url), "http://localhost:4000/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], "Bearer test-key")
                sent = json.loads(request.content)
                self.assertEqual(sent["model"], "proxy-alias")
                prompt = (evaluator.ROOT / "prompts/golden/one-shot.ists.txt").read_text().strip()
                self.assertEqual(sent["messages"], [
                    {"role": "system", "content": prompt}, {"role": "user", "content": pairs[0].message()},
                ])
                self.assertIn("[ both ]", sent["messages"][1]["content"])

            transport = Mock(return_value=completion("malformed"))
            with patch.dict("os.environ", {"OPENAI_API_KEY": ""}), proxy(transport):
                directory = evaluator.run(args)
                self.assertEqual(transport.call_count, 1)
                self.assertNotIn("Authorization", transport.call_args.args[0].headers)
            domain = directory / "answers-students"
            self.assertIn("2", json.loads((domain / "review.json").read_text()))
            with (directory / "results.csv").open() as f:
                row = next(csv.DictReader(f))
                self.assertEqual((row["pairs"], row["invalid"]), ("2", "1"))
                self.assertEqual(row["json_conversion_failed"], "")
                self.assertLess(float(row["f1_ali"]), 1.0)
            (domain / "repairs/2.txt").write_text(pairs[1].gold)
            manifest = directory / "run.json"
            config = json.loads(manifest.read_text())
            config["scorer_sha256"]["penalty"] = "previous-script-hash"
            manifest.write_text(json.dumps(config))
            with proxy(Mock(side_effect=AssertionError("unexpected API call"))):
                evaluator.run(args)
            self.assertEqual(
                json.loads(manifest.read_text())["scorer_sha256"]["penalty"],
                evaluator.digest(evaluator.PENALTY.read_bytes()),
            )
            self.assertEqual(json.loads((domain / "raw/2.json").read_text())["choices"][0]["message"]["content"],
                             "malformed")
            with (directory / "results.csv").open() as f:
                row = next(csv.DictReader(f))
                self.assertEqual((row["invalid"], row["repaired"], row["f1_type_score_penalty"]),
                                 ("0", "1", "1.0000"))

    def test_json_token_indices_and_modifiers(self):
        pair = evaluator.load_pairs("answers-students", "system", 1)[0]
        self.assertNotIn("[", pair.message())
        answer = {"alignments": [
            {"source_ids": [2, 3, 4, 5], "translation_ids": [1, 2, 3, 4], "label": "EQUI_FACT_POL", "score": 5},
            {"source_ids": [], "translation_ids": [6], "label": "NOALI", "score": None},
        ]}
        wa = evaluator.prediction(pair, json.dumps(answer), "jacyna-json")
        self.assertIn("2 3 4 5 <==> 1 2 3 4 // EQUI_FACT_POL // 5", wa)
        self.assertIn("0 <==> 6 // NOALI // NIL", wa)
        answer["alignments"][0]["source_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluator.prediction(pair, json.dumps(answer), "jacyna-json")

    def test_json_conversion_counts_and_repairs(self):
        prediction = {"alignments": [
            {"source_indices": [2, 3, 4, 5], "target_indices": [1, 2, 3, 4], "label": "EQUI", "score": 5},
        ]}
        valid = json.dumps(prediction)
        bad_syntax = '  {"alignments": [\n'
        bad_schema = '{"alignments": [{}]}'
        replies = [completion(valid), completion(bad_syntax), completion(bad_schema)]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stdout:
            args = evaluator.arguments([
                "--model", "proxy-alias", "--prompt", "prompts/system/zero-shot.json.txt",
                "--limit", "3", "--run-dir", directory,
            ])
            transport = Mock(side_effect=replies)
            with proxy(transport):
                run_dir = evaluator.run(args)
            self.assertEqual(transport.call_count, 3)
            self.assertIn("JSON conversion failures: 2/3 (1 with invalid JSON syntax)", stdout.getvalue())
            with (run_dir / "results.csv").open() as f:
                result = next(csv.DictReader(f))
            self.assertEqual((result["json_conversion_failed"], result["malformed_json"]), ("2", "1"))
            domain = run_dir / "answers-students"
            malformed = domain / "malformed"
            self.assertIn(f"Failed replies: {malformed}", stdout.getvalue())
            self.assertEqual(
                {path.name: path.read_text() for path in malformed.iterdir()},
                {"2.txt": bad_syntax, "3.txt": bad_schema},
            )
            annotations = parse_wa((domain / "predictions.wa").read_text())
            self.assertEqual(annotations[0].prediction_dict(), prediction)
            self.assertEqual([item.alignments for item in annotations[1:]], [(), ()])

            repair = domain / "repairs/2.txt"
            shutil.copyfile(malformed / "2.txt", repair)
            repair.write_text(valid)
            with proxy(Mock(side_effect=AssertionError("unexpected API call"))):
                evaluator.run(args)
            with (run_dir / "results.csv").open() as f:
                result = next(csv.DictReader(f))
            self.assertEqual((result["json_conversion_failed"], result["malformed_json"], result["repaired"]),
                             ("1", "0", "1"))
            self.assertEqual((malformed / "2.txt").read_text(), bad_syntax)
            self.assertEqual(set(json.loads((domain / "review.json").read_text())), {"3"})

    def test_canonical_json_prompts_use_the_canonical_schema(self):
        legacy_reply = json.dumps({"alignments": [
            {"source_ids": [1], "translation_ids": [1, 2, 3, 4], "label": "EQUI", "score": 5},
        ]})
        for chunks in ("system", "golden"):
            pair = evaluator.load_pairs("answers-students", chunks, 1)[0]
            payload = parse_wa(pair.gold)[0].prediction_dict()
            for shots, prompt_name in ((0, "zero-shot"), (1, "one-shot")):
                with self.subTest(chunks=chunks, shots=shots), tempfile.TemporaryDirectory() as directory:
                    prompt = evaluator.ROOT / "prompts" / chunks / f"{prompt_name}.json.txt"
                    args = evaluator.arguments([
                        "--model", "proxy-alias", "--prompt", str(prompt), "--limit", "1", "--run-dir", directory,
                    ])
                    response = legacy_reply if shots == 0 else json.dumps(payload)
                    transport = Mock(return_value=completion(response))
                    with proxy(transport), redirect_stdout(io.StringIO()):
                        run_dir = evaluator.run(args)
                    request = json.loads(transport.call_args.args[0].content)
                    self.assertEqual(request["messages"][0]["content"], prompt.read_text().strip())
                    self.assertEqual(request["messages"][1]["content"], pair.message())
                    with (run_dir / "results.csv").open() as f:
                        row = next(csv.DictReader(f))
                    self.assertEqual(row["format"], "json")
                    self.assertEqual(row["json_conversion_failed"], "1" if shots == 0 else "0")
                    if shots == 1:
                        self.assertEqual(row["f1_type_score_no_penalty"], "1.0000")

    def test_renamed_json_formats_resume_with_saved_replies(self):
        reply = completion('{"alignments": []}')
        for old_format, new_format in (("canonical-json", "json"), ("json", "jacyna-json")):
            with self.subTest(format=new_format), tempfile.TemporaryDirectory() as directory:
                args = evaluator.arguments([
                    "--model", "proxy-alias", "--prompt", f"prompts/system/zero-shot.{new_format}.txt",
                    "--limit", "1", "--run-dir", directory,
                ])
                transport = Mock(return_value=reply)
                with proxy(transport), redirect_stdout(io.StringIO()):
                    run_dir = evaluator.run(args)
                request = json.loads(transport.call_args.args[0].content)
                expected = evaluator.ROOT / "prompts" / "system" / f"zero-shot.{new_format}.txt"
                self.assertEqual(request["messages"][0]["content"], expected.read_text().strip())

                manifest = run_dir / "run.json"
                config = json.loads(manifest.read_text())
                old_prompt = f"prompts/system/zero-shot.{old_format}.txt"
                manifest.write_text(json.dumps({**config, "format": old_format, "prompt": old_prompt}))
                with proxy(Mock(side_effect=AssertionError("unexpected API call"))), redirect_stdout(io.StringIO()):
                    evaluator.run(args)
                self.assertEqual(json.loads(manifest.read_text()), config)

    def test_prompt_directory_tracks_calls_and_resumes_a_partial_batch(self):
        """Count only missing calls across prompt directories and preserve cached results."""
        prompt_dir = evaluator.ROOT / "prompts/system"
        prompts = sorted(prompt_dir.glob("*.txt"))
        pair = evaluator.load_pairs("answers-students", "system", 1)[0]
        json_reply = json.dumps(parse_wa(pair.gold)[0].prediction_dict())
        replies = [pair.gold if p.name.endswith(".ists.txt") else json_reply for p in prompts]

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = evaluator.arguments([
                "--model", "proxy-alias", "--prompt-dir", str(prompt_dir),
                "--domain", "answers-students", "--limit", "1", "--run-dir", directory,
            ])
            refused = httpx.Response(401, json={"error": {"message": "bad key"}})
            with proxy(Mock(side_effect=[completion(replies[0]), refused])), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "bad key"):
                    evaluator.run(args)
            self.assertTrue((Path(directory) / prompts[0].stem / "results.csv").is_file())

            transport = Mock(side_effect=[completion(reply) for reply in replies[1:]])
            with proxy(transport), redirect_stderr(io.StringIO()) as progress:
                output = evaluator.run(args)
            remaining = len(prompts) - 1
            self.assertEqual(transport.call_count, remaining)
            self.assertIn(f"{remaining}/{remaining}", progress.getvalue())
            self.assertIn(f"[{len(prompts)}/{len(prompts)}] {prompts[-1].name}", progress.getvalue())
            for call, prompt in zip(transport.call_args_list, prompts[1:]):
                request = json.loads(call.args[0].content)
                self.assertEqual(request["messages"][0]["content"], prompt.read_text().strip())
            for prompt in prompts:
                with (output / prompt.stem / "results.csv").open() as f:
                    self.assertEqual(next(csv.DictReader(f))["f1_type_score_no_penalty"], "1.0000")

    def test_parallel_requests_cache_successes_and_resume_in_pair_order(self):
        """Overlap requests, keep successful replies on error, and resume with a different pool size."""
        pairs = evaluator.load_pairs("answers-students", "system", 2)
        responses = {
            pairs[0].message(): completion(pairs[0].gold),
            pairs[1].message(): httpx.Response(401, json={"error": {"message": "bad key"}}),
        }
        barrier = Barrier(2, timeout=5)

        def proxy_reply(request):
            """Wait for both requests before returning either mocked response."""
            message = json.loads(request.content)["messages"][1]["content"]
            barrier.wait()
            return responses[message]

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            argv = [
                "--model", "proxy-alias", "--prompt", "prompts/system/zero-shot.ists.txt",
                "--limit", "2", "--run-dir", directory,
            ]
            with proxy(proxy_reply), self.assertRaisesRegex(RuntimeError, "answers-students pair 2"):
                evaluator.run(evaluator.arguments([*argv, "--workers", "2"]))
            domain = Path(directory) / "answers-students"
            self.assertEqual([path.name for path in (domain / "raw").iterdir()], ["1.json"])

            transport = Mock(return_value=completion(pairs[1].gold))
            with proxy(transport):
                evaluator.run(evaluator.arguments([*argv, "--workers", "1"]))
            self.assertEqual(transport.call_count, 1)
            request = json.loads(transport.call_args.args[0].content)
            self.assertEqual(request["messages"][1]["content"], pairs[1].message())
            predictions = parse_wa((domain / "predictions.wa").read_text())
            self.assertEqual([annotation.context.sentence_id for annotation in predictions], ["1", "2"])

    def test_penalty_can_earn_credit_when_type_f1_is_zero(self):
        pair = evaluator.load_pairs("answers-students", "system", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.wa"
            payload = {"alignments": [
                {"source_indices": [7], "target_indices": [5], "label": "SIMI", "score": 4},
            ]}
            path.write_text(json_to_wa(pair.context(), payload) + "\n")
            metrics = evaluator.score([pair], path)
            self.assertEqual(metrics["f1_type_score_no_penalty"], "0.0000")
            self.assertGreater(float(metrics["f1_type_score_penalty"]), 0)
            path.write_text(json_to_wa(pair.context(), {"alignments": []}) + "\n")
            metrics = evaluator.score([pair], path)
            self.assertEqual(metrics["f1_type_score_penalty"], "0.0000")
