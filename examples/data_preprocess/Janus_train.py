# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/math",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = [
        # "/mnt/jfzn/yhc/datasets/verl/dapo-math-17k.parquet",
        "/mnt/jfzn/yhc/datasets/hkust-nlp/SimpleRL-Zoo-Data/simplelr_abel_gsm8k_level1/train.parquet",
        "/mnt/jfzn/yhc/datasets/hkust-nlp/SimpleRL-Zoo-Data/simplelr_abel_level1to4/train.parquet",
        "/mnt/jfzn/yhc/datasets/hkust-nlp/SimpleRL-Zoo-Data/simplelr_abel_level3to5/train.parquet",
    ]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question: str = example.pop("prompt")[0]["content"]

            answer = example.pop("reward_model")["ground_truth"]
            # solution = extract_solution(answer)
            data = {
                "data_source": "math_simplerl",
                "prompt": [
                    # {"role": "system", "content": instruction_following},
                    # {"role": "user", "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"+question.replace("Answer:\nLet's think step by step.", "")+'Remember to put your answer on its own line after \"Answer:\".'},
                    {
                        "role": "user",
                        # "content": question.replace(
                        #     "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
                        #     "",
                        # ).replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
                        # + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
                        "content": question.replace(
                            "\nAnswer:\nLet's think step by step.\n",
                            "",
                        )
                        + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    merge_dataset = None
    for local_dataset in local_dataset_path:
        dataset = datasets.load_dataset("parquet", data_files=local_dataset)
        # If load_dataset returns a DatasetDict (e.g. with a 'train' split), pick the first split.
        if hasattr(dataset, "keys"):
            first_split = next(iter(dataset.keys()))
            dataset = dataset[first_split]
        train_dataset = dataset.map(function=make_map_fn("train"), with_indices=True)
        if merge_dataset is None:
            merge_dataset = train_dataset
        else:
            merge_dataset = datasets.concatenate_datasets([merge_dataset, train_dataset])
    merge_dataset.to_parquet(os.path.join(args.local_save_dir, "train_simplerl.parquet"))
