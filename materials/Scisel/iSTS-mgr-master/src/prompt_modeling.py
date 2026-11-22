from src.constants import INSTRUCTIONS, LABELS_DESCRIPTION

def create_scoring_query(text1, text2, label):
    query = f"Chunk 1: [{text1}], Chunk 2: [{text2}], Label: {label}"

    return query

def create_chunking_query(raw_sentence, chunked_sentence):
    query = f"Raw: {raw_sentence} \n" \
            f"Chunked: {chunked_sentence}"
    
    return query


def create_label(row, task_type):
    if task_type in ["score", "alignment"]:
        return row[task_type]
    elif task_type == "merged":
        return f"{row["score"]}-{row["alignment"]}"
    elif task_type == "both":
        return f"ALIGNMENT: {row["alignment"]}; SCORE: {row["score"]}"
    else:
        raise ValueError(f"wrong task_type: {task_type}")
    

def create_prompt(row, df, task_type, instruction_type=None, label_description_type=None, prompt_type='eval', n_examples=0):

    sampled_examples_df = (
        df.sample(frac=1, random_state=42)
        .groupby([task_type], sort=False)
        .head(n_examples)
        .reset_index(drop=True)
    )
    prompt_parts = []
    if instruction_type is not None:
        prompt_parts.append(INSTRUCTIONS[instruction_type])
    if label_description_type is not None:
        prompt_parts.append(LABELS_DESCRIPTION[task_type][label_description_type])

    examples = []
    if n_examples > 0:
        prompt_parts.append("Examples:")
        for index, sampled_row in sampled_examples_df.iterrows():
            example = f"{index}) Chunk 1: [{sampled_row['text_source']}], Chunk 2: [{sampled_row['text_translation']}], Label: {create_label(sampled_row, task_type)}"
            examples.append(example)
    examples_str = "\n".join(examples)
    prompt_parts.append(examples_str)

    query = f"Chunks to label: Chunk 1: [{row['text_source']}], Chunk 2: [{row['text_translation']}], Label:{create_label(row, task_type) if prompt_type == 'train' else ''}"
    prompt_parts.append(query)
    
    prompt = "\n\n".join(prompt_parts)

    return prompt


def create_evaluation_prompt(row, task_type, instruction_type=None, label_description_type=None):

    prompt_parts = []
    if instruction_type is not None:
        prompt_parts.append(INSTRUCTIONS[instruction_type])
    if label_description_type is not None:
        prompt_parts.append(LABELS_DESCRIPTION[task_type][label_description_type])

    query = f"Chunk 1: [{row['text_source']}], Chunk 2: [{row['text_translation']}]"

    prompt_parts.append(query)
    prompt = "\n".join(prompt_parts)

    return prompt    



def create_train_prompt(row, df, task_type, instruction_type=None, label_description_type=None, n_examples=0):

    prompt_parts = []
    if instruction_type is not None:
        prompt_parts.append(INSTRUCTIONS[instruction_type])
    if label_description_type is not None:
        prompt_parts.append(LABELS_DESCRIPTION[task_type][label_description_type])

    query = f"Chunk 1: [{row['text_source']}], Chunk 2: [{row['text_translation']}]"

    prompt_parts.append(query)
    prompt = "\n".join(prompt_parts)

    completion = f"Label: {create_label(row, task_type)}"

    return prompt, completion
