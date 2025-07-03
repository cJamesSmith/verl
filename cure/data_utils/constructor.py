from typing import Dict, List

import pandas as pd
from jinja2 import Template
from numpy import random
from transformers import AutoTokenizer

from cure.data_utils.prompts import special_requirements, system_case_prompts, system_prompts


def get_code_case_data(
    data: List[Dict],
    target_data_len: int,
    # content_max_length: int,
    code_parquet_path: str,
    case_parquet_path: str,
    split: str,
    tokenizer: AutoTokenizer,
):
    return_io_data = []

    probabilities = [1.0 / len(data)] * len(data)

    if target_data_len is not None:
        # Randomly select target_data_len entries from io_data
        if len(data) >= target_data_len:
            selected_indices = random.choice(len(data), size=target_data_len, replace=False, p=probabilities)
        else:
            # If not enough data, upsample with replacement
            selected_indices = random.choice(len(data), size=target_data_len, replace=True, p=probabilities)
    else:
        # If target_data_len is None, use all data
        selected_indices = list(range(len(data)))

    # Generate code parquet
    for idx in selected_indices:
        io_prompt = Template(system_prompts).render(
            language="python",
            special_requirements=special_requirements,
            problem=data[idx]["question"],
        )
        # assert len(tokenizer(io_prompt)["input_ids"]) <= content_max_length
        io_item = {
            "data_source": data[idx]["dataset"],
            "prompt": [
                {
                    "role": "user",
                    "content": io_prompt,
                }
            ],
            "problem": "",
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": "",
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "metric": "code",
            },
        }
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(code_parquet_path)

    # Generate case parquet
    return_io_data = []
    for idx in selected_indices:
        n_example = len(data[idx]["example_input"])
        example_input = ", ".join([repr(item) for item in data[idx]["example_input"]])
        example_output = ", ".join([repr(item) for item in data[idx]["example_output"]])
        if n_example == 0:
            example_intro = """ """
        if n_example == 1:
            example_intro = """We already have one test sample:\n Its input is {{example_input}}. Its output is {{example_output}}.\n"""
            example_intro = Template(example_intro).render(example_input=example_input, example_output=example_output)
        if n_example > 1:
            example_intro = """We already have {{n_sample}} test samples:\n The inputs are, respectively, {{example_input}}. The corresponding outputs are {{example_output}}.\n"""
            example_intro = Template(example_intro).render(n_sample=n_example, example_input=example_input, example_output=example_output)

        io_prompt = Template(system_case_prompts).render(problem=data[idx]["question"], example_intro=example_intro)

        # assert len(tokenizer(io_prompt)['input_ids']) <= content_max_length
        io_item = {
            "data_source": data[idx]["dataset"],
            "prompt": [
                {
                    "role": "user",
                    "content": io_prompt,
                }
            ],
            "problem": "",
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": "",
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "metric": "case",
            },
        }
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(case_parquet_path)

    return [data[idx] for idx in selected_indices]
