from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch
import os
import time
from pathlib import Path
import textwrap

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

#  CONFIG:
SENT_FILE_1 = "1.txt"
SENT_FILE_2 = "2.txt"
OUTPUT_TXT  = "out.txt"



def print_vram(tag: str = ""):
    """Print VRAM usage for all visible GPUs."""
    if not torch.cuda.is_available():
        print(f"[VRAM] {tag}: CUDA not available", flush=True)
        return

    gb = 1024 ** 3
    print(f"\n[VRAM] {tag}", flush=True)
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)  # bytes
        used = total - free
        print(
            f" GPU {i}: used {used / gb:.2f} GB / {total / gb:.2f} GB "
            f"({used / total * 100:.1f}%)",
            flush=True,
        )


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


def build_prompt(s1: str, s2: str, idx) -> str:
    instructions = textwrap.dedent(
        """
             PROMPT #TUTAJ WKLEIC PROMPT

        """
    ).strip()

    pair_text = textwrap.dedent(
        f"""
        Sentence 1:
        {s1}

        Sentence 2:
        {s2}

        Now analyse these two sentences according to the iSTS instructions. Current sentence id is: {idx}
        """
    ).strip()

    return instructions + "\n\n" + pair_text


def main():
    # Use only HF_TOKEN from env
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Export it before running:\n"
            '  export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"'
        )

    print_vram("before loading processor/model")

    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)

    print("Loading model (first time = slow, downloads to hf-cache)...", flush=True)
    t0 = time.time()
    model = Llama4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager", 
        token=token,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f} s", flush=True)

    print_vram("after model load")

    # Show how HF spread the model across devices
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is not None:
        print("\n=== HF DEVICE MAP ===", flush=True)
        for module_name, dev in hf_device_map.items():
            print(f" {module_name} -> {dev}", flush=True)
    else:
        print("\n[INFO] model.hf_device_map not available", flush=True)


    print("Loading sentences from text files...", flush=True)
    sents1, sents2 = load_sentences(SENT_FILE_1, SENT_FILE_2)
    num_pairs = len(sents1)
    print(f"Loaded {num_pairs} sentence pairs.", flush=True)

    out_path = Path(OUTPUT_TXT)
    print(f"Writing model outputs to: {out_path.resolve()}", flush=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, (s1, s2) in enumerate(zip(sents1, sents2), start=1):
            print(f"[{idx}/{num_pairs}] Processing pair...", flush=True)

            prompt = build_prompt(s1, s2,idx)

            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ]

            print("Tokenizing via chat template...", flush=True)
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            print_vram(f"before generation (pair {idx})")
            print("Starting generation...", flush=True)

            t1 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                )
            print(f"Generation finished in {time.time() - t1:.1f} s", flush=True)

            print_vram(f"after generation (pair {idx})")

            
            generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
            answer = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            #out_f.write(f"### PAIR {idx} ###\n")
            #out_f.write("Sentence 1:\n")
            #out_f.write(s1 + "\n\n")
            #out_f.write("Sentence 2:\n")
            #out_f.write(s2 + "\n\n")
            #out_f.write("Model answer:\n")
            out_f.write(answer + "\n \n")
            #out_f.write("\n" + "#" * 50 + "\n\n")

            print("ANSWER TO PROMPT:\n", flush=True)
            print(answer + "\n", flush=True)

    print("Done", flush=True)


if __name__ == "__main__":
    main()
