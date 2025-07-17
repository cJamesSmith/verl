import os
from functools import partial
import subprocess
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re
import uuid
from functools import partial
import io
import os
import sys
import ast
import json
import time
import argparse
import numpy as np
import multiprocessing

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output

def modify(c):
    # Remove any occurrences of "plaintext\n"
    c = c.replace("plaintext\n", "")
    
    # Convert literal "\n" to actual newlines
    c = c.replace("\\n", "\n")
    
    # Ensure there's a trailing newline
    if not c.endswith("\n"):
        c += "\n"
    
    return c

def extract_test_cases(full_output):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r'\*\*Test Input:\*\*\s*```input(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```output(.*?)```'
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    fail_case = [""]
    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        test_input = [modify(matches_input[-1].lstrip('\n'))]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
        pattern_input_plain = r'\*\*Test Input:\*\*\s*```(.*?)```'
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        if matches_input_plain:
            test_input = [modify(matches_input_plain[-1].strip())]
        else:
            test_input = fail_case
    
    # For Test Output: either use the updated triple-backtick version or fallback to plain text
    if matches_output:
        test_output = [modify(matches_output[-1].lstrip('\n'))]
    else:
        # Fallback: capture until the **Explanation:** marker or end-of-string
        pattern_output_plain = r'\*\*Test Output:\*\*\s*```(.*?)```'
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        if matches_output_plain:
            test_output = [modify(matches_output_plain[-1].strip())]
        else:
            test_output = fail_case
    
    # Also extract from the last occurrence of **Test Input:** to the end
    index = full_output.rfind("**Test Input:**")
    if index != -1:
        example_text = [full_output[index:]]
    else:
        example_text = fail_case
    
    # If any essential piece is missing, return empties
    if example_text == fail_case or test_input == fail_case or test_output == fail_case:
        return fail_case, fail_case, fail_case
    
    return test_input, test_output, example_text

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def normalize_reward(reward_arr):
    if np.all(reward_arr == 1):
        return reward_arr
    mean = np.mean(reward_arr)
    std = np.std(reward_arr)
    if std.item() == 0:
        return None
    return (reward_arr - mean) / std

def normalize_balance_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    sum_pos = x[pos_mask].sum()
    sum_neg_abs = abs(x[neg_mask].sum())
    if sum_pos * sum_neg_abs == 0:
        return None
    scale_factor = sum_neg_abs / sum_pos
    x[pos_mask] *= scale_factor
    return x / x.std()

def length_regularize(reward_arr, response_length_list):
    reward_arr = np.sign(reward_arr)
    pos_list = np.where(reward_arr == 1)[0].tolist()
    neg_list = np.where(reward_arr == -1)[0].tolist()
    pos_response_length = np.array([response_length_list[j] for j in pos_list])
    threshold = np.median(pos_response_length).item()
    if np.sum((pos_response_length - threshold)**2) == 0: # no variance
        return normalize_balance_std(np.sign(reward_arr))
    threshold = max(min(threshold, 8000), 1000)
    length_reg_reward = np.zeros(len(reward_arr), float)
    length_reg_reward[pos_list] = - pos_response_length + threshold
    length_reg_reward[neg_list] = np.min(length_reg_reward).copy()
    length_reg_reward = normalize_balance_std(length_reg_reward)
    return length_reg_reward

def code_reward_fn(
        data: List[DataProto],
        tokenizer: AutoTokenizer,
        n_samples: int = 16,
        max_ground_truth_test: int = 8,
        step: int = 0,
        train: bool = True
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """We will expand this function gradually based on the available datasets"""

        code_batch, case_batch, selected_data = data

        # preprocess
        for data in selected_data:
            data["full_code_generation"] = []
            data["code_response_length"] = []
            data["full_case_generation"] = []
            data["case_response_length"] = []
            data["generated_code"] = []
            max_k = min(max_ground_truth_test, len(data["test_input"]))
            data["num_ground_truth_test"] = max_k
            data["all_case_input"] = (data["test_input"][:max_k]).copy()
            data["all_case_output"] = (data["test_output"][:max_k]).copy()
            data["case_input"] = []
            data["case_output"] = []
            data["case_text"] = []

        # process generated codes
        for idx, code_b in enumerate(code_batch):
            code_text = tokenizer.decode(code_b.batch['responses'], skip_special_tokens=True)
            code_output = extract_code(code_text)
            selected_data[idx//n_samples]["code_prompt"] = tokenizer.decode(_pre_process_inputs(tokenizer.pad_token_id, code_b.batch['prompts']), skip_special_tokens=False)
            selected_data[idx//n_samples]['full_code_generation'] = selected_data[idx//n_samples]['full_code_generation'] + [code_text]
            selected_data[idx//n_samples]['generated_code'] = selected_data[idx//n_samples]['generated_code'] + [code_output]
            selected_data[idx//n_samples]['code_response_length'].append(len(tokenizer.encode(code_text, add_special_tokens=False)))

        # process generated unit tests
        for idx, case_b in enumerate(case_batch):
            case_text = tokenizer.decode(case_b.batch['responses'], skip_special_tokens=True)
            test_input, test_output, example_text = extract_test_cases(case_text)
            case_prompt = tokenizer.decode(_pre_process_inputs(tokenizer.pad_token_id, case_b.batch['prompts']), skip_special_tokens=False)
            selected_data[idx // n_samples]["case_prompt"] = case_prompt
            selected_data[idx // n_samples]["full_case_generation"] = selected_data[idx // n_samples]["full_case_generation"] + [case_text]
            selected_data[idx // n_samples]["case_input"] = selected_data[idx // n_samples]["case_input"] + test_input
            selected_data[idx // n_samples]["case_output"] = selected_data[idx // n_samples]["case_output"] + test_output
            selected_data[idx // n_samples]["case_text"] = selected_data[idx // n_samples]["case_text"] + example_text
            selected_data[idx // n_samples]["all_case_input"] = selected_data[idx // n_samples]["all_case_input"] + test_input
            selected_data[idx // n_samples]["all_case_output"] = selected_data[idx // n_samples]["all_case_output"] + test_output
            selected_data[idx // n_samples]["case_response_length"].append(len(tokenizer.encode(case_text, add_special_tokens=False)))

        if train:
            with open("./outputs-execute.json", "w", encoding="utf-8") as f:
                json.dump(selected_data, f, indent=2, ensure_ascii=False)
        else:
            with open("./outputs-execute-test.json", "w", encoding="utf-8") as f:
                json.dump(selected_data, f, indent=2, ensure_ascii=False)

        subprocess.run(
            f'python cure/rewards/exe.py {"--train" if train else ""}',
            shell=True,
            check=True,
        )  # TODO: use fusion sandbox

        if train:
            with open("./outputs-execute.json") as f:
                data = json.load(f)
        else:
            return

        code_reward_tensor = torch.zeros_like(code_batch.batch['responses'], dtype=torch.float32)
        case_reward_tensor = torch.zeros_like(case_batch.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            if data[i]["all_case_bool_table"] is None:
                continue
            
            t = data[i]["num_ground_truth_test"]
            all_test_table_i = np.array(data[i]["all_case_bool_table"])[:, :t].copy()
            all_case_table_i = np.array(data[i]["all_case_bool_table"])[:, t:].copy()

            # reward for code
            code_reward = np.mean(all_test_table_i, 1)
            code_reward = normalize_reward(code_reward)
            if code_reward is not None:
                code_reward = length_regularize(code_reward, data[i]["code_response_length"])
                if code_reward is not None:
                    code_reward = code_reward.tolist()
                    for j in range(len(code_reward)):
                        prompt_ids = code_batch[i*n_samples + j].batch['prompts']
                        prompt_length = prompt_ids.shape[-1]
                        valid_response_length = code_batch[i*n_samples + j].batch['attention_mask'][prompt_length:].sum()
                        code_reward_tensor[i*n_samples + j, valid_response_length - 1] = code_reward[j]
            
            # reward for case
            correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
            if len(correct_code_list) > 0:
                # get reward sign
                correct_code_table = all_case_table_i[correct_code_list, :].copy()
                index_list = np.where(np.all(correct_code_table, axis=0))[0].tolist()
                reward_sign = -np.ones(correct_code_table.shape[1], dtype=float)
                reward_sign[index_list] = 1
                case_reward = reward_sign.copy()
                # get reward scale
                wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
                if len(wrong_code_list) > 0:
                    reward_scale = np.ones(correct_code_table.shape[1], dtype=float)
                    correct_case_list = np.where(correct_code_table.all(axis=0))[0].tolist()
                    wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
                    if len(correct_case_list):
                        wrong_code_correct_case_table = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
                        if step < 100:  # TODO: True by default, but suggest False after 100+ optimization steps
                            mean_p01 = np.mean(~wrong_code_correct_case_table, 0)
                        else:
                            mean_p01 = (~np.any(wrong_code_correct_case_table, axis=0)).astype(float)
                        reward_scale[correct_case_list] = reward_scale[correct_case_list] * mean_p01
                    if len(wrong_case_list):
                        wrong_code_wrong_case_table = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
                        if step < 100:  # TODO: True by default, but suggest False after 100+ optimization steps
                            mean_p00 = np.mean(wrong_code_wrong_case_table, 0)
                        else:
                            mean_p00 = (np.any(wrong_code_wrong_case_table, axis=0)).astype(float)
                        reward_scale[wrong_case_list] = reward_scale[wrong_case_list] * mean_p00
                    case_reward = case_reward * reward_scale
                
                case_reward = normalize_reward(case_reward)
                if case_reward is not None:
                    case_reward = length_regularize(case_reward, data[i]["case_response_length"])
                    if case_reward is not None:
                        case_reward = case_reward.tolist()
                        for j in range(len(case_reward)):
                            prompt_ids = case_batch[i*n_samples + j].batch['prompts']
                            prompt_length = prompt_ids.shape[-1]
                            valid_response_length = case_batch[i*n_samples + j].batch['attention_mask'][prompt_length:].sum()
                            case_reward_tensor[i*n_samples + j, valid_response_length - 1] = case_reward[j]

        return code_reward_tensor, case_reward_tensor
