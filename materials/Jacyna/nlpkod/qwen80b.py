from vllm import LLM, SamplingParams
import torch
import os
from pathlib import Path
import textwrap
import subprocess

MODEL_ID = "cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"

#  CONFIG:
SENT_FILE_1 = "1.txt"
SENT_FILE_2 = "2.txt"
OUTPUT_TXT  = "out.txt"

MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN = 5128
GPU_UTIL        = 0.7



def print_gpu_usage():
    """Print total VRAM usage across all visible GPUs."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        mem_list = [int(x) for x in result.decode().splitlines()]
        total = sum(mem_list)
        print(f"[GPU VRAM] Total used across all GPUs: {total} MB", flush=True)
    except Exception as e:
        print(f"[GPU VRAM] Could not get GPU memory usage: {e}", flush=True)


def load_sentences(path1: str, path2: str):
    p1, p2 = Path(path1), Path(path2)

    if not p1.exists():
        raise FileNotFoundError(f"File not found: {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"File not found: {p2}")

    with p1.open("r", encoding="utf-8") as f1:
        sents1 = [line.strip() for line in f1 if line.strip()]
    with p2.open("r", encoding="utf-8") as f2:
        sents2 = [line.strip() for line in f2 if line.strip()]

    if len(sents1) != len(sents2):
        raise ValueError(
            f"Files have different number of non-empty lines: "
            f"{len(sents1)} vs {len(sents2)}"
        )

    return sents1, sents2


def build_prompt(s1: str, s2: str,idx) -> str:
    instructions = textwrap.dedent(
        """
        PROMPT #TUTAJ PROMPT
        """
    ).strip()

    pair_text = textwrap.dedent(
        f"""
        Sentence 1:
        {s1}

        Sentence 2:
        {s2}

         Now analyse these two sentences according to the iSTS instructions. Current id is: {idx}. put this id exaclly as the sentence id. only this number as this id
        """
    ).strip()

    return instructions + "\n\n" + pair_text


def load_model_and_tokenizer():
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Visible GPUs: {num_gpus}", flush=True)

    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs visible.")

    print(f"Loading vLLM engine on {num_gpus} GPUs...", flush=True)

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=num_gpus,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_UTIL,
        trust_remote_code=True,
        enforce_eager=True,   # disables torch.compile (prevents nvcc error)
    )

    tokenizer = llm.get_tokenizer()

    print_gpu_usage()   # Print VRAM after model load

    return tokenizer, llm


def generate_freeform_answer(tokenizer, llm, prompt: str) -> str:
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.3,
       
    )

    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text

    return text


def main():
    print("Loading sentences from text files...", flush=True)
    sents1, sents2 = load_sentences(SENT_FILE_1, SENT_FILE_2)
    num_pairs = len(sents1)
    print(f"Loaded {num_pairs} sentence pairs.", flush=True)

    tokenizer, llm = load_model_and_tokenizer()

    out_path = Path(OUTPUT_TXT)
    print(f"Writing free-form model outputs to: {out_path.resolve()}", flush=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, (s1, s2) in enumerate(zip(sents1, sents2), start=1):
            print(f"[{idx}/{num_pairs}] Processing pair...", flush=True)

            prompt = build_prompt(s1, s2,idx)
            answer = generate_freeform_answer(tokenizer, llm, prompt)

            print_gpu_usage()   # Print VRAM after each prompt

            #out_f.write(f"### PAIR {idx} ###\n")
            #out_f.write("Sentence 1:\n")
            #out_f.write(s1 + "\n\n")
            #out_f.write("Sentence 2:\n")
            #out_f.write(s2 + "\n\n")
            #out_f.write("Model answer:\n")
            out_f.write(answer + "\n \n")
            #out_f.write("\n" + "#" * 50 + "\n\n")
            
            print("ANSWER TO PROMPT: \n", flush=True)
            print(answer + "\n", flush=True)

    print("Done", flush=True)


if __name__ == "__main__":
    main()
