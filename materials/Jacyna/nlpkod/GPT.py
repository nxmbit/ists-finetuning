from openai import OpenAI
print("dziala")

import os
from pathlib import Path
import textwrap

# CONFIG:
SENT_FILE_1 = "sent1.txt"          
SENT_FILE_2 = "sent2.txt"          
OUTPUT_TXT  = "out.txt"



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



instructions = textwrap.dedent(
    """
    PROMPT  #TUTAJ TRZEBA WKLEIC PROMPT

    """
).strip()



client = OpenAI(api_key='WKLEJ KLUCZ API', # wklej klucz api
    
    timeout=900.0
)



def main():
    print("Loading sentences from text files...", flush=True)
    sents1, sents2 = load_sentences(SENT_FILE_1, SENT_FILE_2)
    num_pairs = len(sents1)
    print(f"Loaded {num_pairs} sentence pairs.", flush=True)


    out_path = Path(OUTPUT_TXT)
    print(f"Writing free-form model outputs to: {out_path.resolve()}", flush=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, (s1, s2) in enumerate(zip(sents1, sents2), start=1):
            print(f"[{idx}/{num_pairs}] Processing pair...", flush=True)

            pair_text = textwrap.dedent(
                f"""
                Sentence 1:
                {s1}

                Sentence 2:
                {s2}

                Now analyse these two sentences according to the iSTS instructions. Current sentence id is: {idx}
                """
            ).strip()

            response = client.with_options(timeout=900.0).responses.create(
            model="gpt-5.1",
            instructions=instructions,
            input=pair_text,
            service_tier="flex",
            )

            print("chat responce to prompt: \n", flush=True)
            print(response.output_text)
        

            #out_f.write(f"### PAIR {idx} ###\n")
            #out_f.write("Sentence 1:\n")
            #out_f.write(s1 + "\n\n")
            #out_f.write("Sentence 2:\n")
            #out_f.write(s2 + "\n\n")
            #out_f.write("Model answer:\n")
            out_f.write(response.output_text + "\n \n" )
            #out_f.write("\n" + "#" * 50 + "\n\n")

    print("Done", flush=True)


if __name__ == "__main__":
    main()