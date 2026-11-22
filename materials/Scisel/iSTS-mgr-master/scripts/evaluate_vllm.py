from src.utils import select_label_from_response
import os
import time
import argparse
from omegaconf import OmegaConf
from transformers import set_seed
from src.prompt_modeling import create_prompt
from src.data.data import get_dataframe
from functools import partial
from src.evaluation.evaluate_vllm import VllmEvaluator
from src.constants import VLLM_SYSTEM_PROMPT


def main(dataset, n_examples, description_level, task):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)

    args = parser.parse_args()

    
    START_TIME = time.strftime("%Y%m%d_%H%M%S")

    cfg = OmegaConf.load(args.config_file)
    
    set_seed(cfg.random_seed)
    
    df_test_wa = get_dataframe(dataset, cfg.test_subset, cfg.source)
    df_train_wa = get_dataframe(dataset, cfg.train_subset, cfg.source)
    
    _formating_prompt = partial(create_prompt, df=df_train_wa, 
                                instruction_type=task,
                                label_description_type=description_level, 
                                task_type=task,
                                n_examples = n_examples)
    
    prompts = df_test_wa.apply(_formating_prompt, axis=1)

    evaluator = VllmEvaluator(cfg.model_name, VLLM_SYSTEM_PROMPT, output_dir="./vllm")

    responses = evaluator(prompts)

    preds = [select_label_from_response(response, task) for response in responses]
    
    result_df = df_test_wa
    result_df["preds"] = preds
    result_df["response"]= responses
    

    save_dir = (
        "results_new/"
        f"{cfg.task_adaptation}/"
        f"{task}/"
        f"{cfg.model_name}/"
        f"{dataset}_"
        f"{description_level}_"
        f"{n_examples}_"
        f"{START_TIME}"
    )
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "results.csv"), index=False)

if __name__ == "__main__":
    datasets = ["headlines", "answers-students", "images"]
    levels = ["simple", "detailed"]
    n_examples = [0, 1, 3, 5]
    tasks = ["merged", "both"]
    for task in tasks:
        for dataset in datasets:
            for level in levels:
                for n_example in n_examples:
                    main(dataset, n_example, level, task)