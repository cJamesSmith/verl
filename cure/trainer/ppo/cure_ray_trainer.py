import gc
import json
import os
import threading
import uuid
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import numpy as np
import ray
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from cure.data_utils.constructor import get_code_case_data
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)


@ray.remote
class DatasetManager:
    def __init__(self):
        self.datasets = {
            "train": [],
        }
        self.locks = {
            "train": threading.Lock(),
        }

    def update_train_data(self, entries):
        with self.locks["train"]:
            self.datasets["train"].extend(entries)
            return len(entries)

    def get_dataset(self, name) -> List[Dict]:
        return self.datasets[name]


class CURERayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_manager = DatasetManager.remote()

    def _create_train_code_case_dataloader(self, data_len: int) -> DataLoader:
        train_dataset = ray.get(self.dataset_manager.get_dataset.remote("train"))

        # code dataloader TODO: directly use code data to build RLHFDataset
        code_parquet_path = (Path(self.config.trainer.default_local_dir) / "code" / "train_code.parquet").as_posix()
        case_parquet_path = (Path(self.config.trainer.default_local_dir) / "code" / "train_case.parquet").as_posix()
        os.makedirs(os.path.dirname(code_parquet_path), exist_ok=True)

        # Common parameters for get_code_io_data
        gen_params = {
            "data": train_dataset,
            "target_data_len": data_len,
            # "content_max_length": self.config.cure.content_max_length,
            "code_parquet_path": code_parquet_path,
            "case_parquet_path": case_parquet_path,
            "split": "train",
            "tokenizer": self.tokenizer,
        }

        selected_data = get_code_case_data(**gen_params)

        code_train_dataset = RLHFDataset(
            data_files=code_parquet_path,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )
        case_train_dataset = RLHFDataset(
            data_files=case_parquet_path,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )

        code_sampler = SequentialSampler(code_train_dataset)
        case_sampler = SequentialSampler(case_train_dataset)

        return (
            iter(DataLoader(dataset=code_train_dataset, batch_size=self.config.data.train_batch_size, drop_last=True, collate_fn=collate_fn, sampler=code_sampler)),
            iter(DataLoader(dataset=case_train_dataset, batch_size=self.config.data.train_batch_size, drop_last=True, collate_fn=collate_fn, sampler=case_sampler)),
            selected_data,
        )

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Overide the _create_dataloader method, we will create the dataloader in the fit method
        """
        return None

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # TODO: perform validation before training

        # load the training dataset
        with open(self.config.cure.train_files) as file:
            train_dataset = json.load(file)  # for *.json file
        ray.get(self.dataset_manager.update_train_data.remote(train_dataset))

        # add tqdm
        progress_bar = tqdm(total=self.config.cure.total_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        while self.global_steps < self.config.cure.total_steps:
            data_len = self.config.data.train_batch_size * 2 - 1  # TODO: prevent the last batch from being too small, be carefull
            code_dataloader, case_dataloader, selected_data = self._create_train_code_case_dataloader(data_len=data_len)

            metrics = {}
            timing_raw = {}

            # 1. Handle code genration, we only need the first batch (because we previously sample more than a batch to avoid data length issues)
            code_batch_dict = next(code_dataloader)
            case_batch_dict = next(case_dataloader)
            code_batch: DataProto = DataProto.from_single_dict(code_batch_dict)
            case_batch: DataProto = DataProto.from_single_dict(case_batch_dict)


            batch: DataProto = DataProto.concat([code_batch, case_batch])  # TODO: we need to check if concat is right
            print(f"len(batch): {len(batch)}")
            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])
            gen_batch.meta_info["do_sample"] = True

            is_last_step = self.global_steps >= self.config.cure.total_steps


            with _timer("step", timing_raw):
                # 1.1 Generate a batch
                with _timer('gen', timing_raw):
                    print("Starting generation")
                    gen_batch_output_list = []
                    # breakpoint()
                    for i in range(len(gen_batch) // 8):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch[i*8:(i+1)*8])
                        gen_batch_output_list.append(gen_batch_output)
                        print(len(gen_batch_output_list))
                    gen_batch_output = DataProto.concat(gen_batch_output_list)
                    del gen_batch_output_list
                    gc.collect()
                    # if not self.async_rollout_mode:
                    #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    # else:
                    #     self.async_rollout_manager.wake_up()
                    #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                    #     self.async_rollout_manager.sleep()
                    print("Generation finished")
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                batch.batch["response_mask"] = compute_response_mask(batch)

                code_batch, case_batch = batch.chunk(2)
                reward_fn_kwargs = {
                    'data': [code_batch, case_batch, selected_data],
                    'tokenizer': self.tokenizer,
                    'n_samples': self.config.actor_rollout_ref.rollout.n,
                    'max_ground_truth_test': self.config.cure.sample.max_ground_truth_test,
                    'step': self.global_steps,
                }
                with _timer("reward_fn", timing_raw):
                    code_reward_tensor, case_reward_tensor = self.reward_fn(**reward_fn_kwargs)
                
                code_batch.batch['token_level_scores'] = code_reward_tensor
                case_batch.batch['token_level_scores'] = case_reward_tensor

                batch: DataProto = DataProto.concat([code_batch, case_batch])

                token_level_scores = batch.batch['token_level_scores']

                # idx = torch.where(token_level_scores.abs().sum(-1).view(-1, self.config.actor_rollout_ref.rollout.n).sum(-1))[0]
                # if len(idx) == 0:
                #     continue
                # idx = torch.repeat_interleave(idx, self.config.actor_rollout_ref.rollout.n)
                # idx *= self.config.actor_rollout_ref.rollout.n  # align with repeated responses in rollout
                # batch = batch[idx]  # As the paper's original implementation, we only compute adv for non-zero token_level_scores

                idx = torch.where(token_level_scores.abs().sum(-1)!=0)[0]
                while len(idx) < self.actor_rollout_wg.world_size:
                    idx = torch.cat([idx, idx])

                batch = batch[idx]

                batch_size = len(batch)
                batch_size = batch_size // self.actor_rollout_wg.world_size * self.actor_rollout_wg.world_size
                batch = batch[:batch_size]  # make sure the batch size is divisible by world size

                # We DO NOT use balance_batch, because it will cause issue when calculate reward
                # balence batch and reorder batch
                # breakpoint()
                self._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with _timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # We DO NOT use critics, we use reward function instead

                code_batch, case_batch = batch.chunk(2)

                with _timer("adv", timing_raw):
                    print("Start computing rewards and advantages")
                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    # norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                    assert self.config.algorithm.adv_estimator == 'grpo', "Currently only GRPO is supported for CURE PPO, please set adv_estimator to 'grpo' in your config."
                    scores = batch.batch["token_level_rewards"].sum(dim=-1)
                    scores = scores.unsqueeze(-1) * batch.batch["response_mask"]
                    batch.batch["advantages"] = scores
                    batch.batch["returns"] = scores
                    print("Rewards and advantages computed successfully")
                    # batch = compute_advantage(
                    #         batch,
                    #         adv_estimator=self.config.algorithm.adv_estimator,
                    #         gamma=self.config.algorithm.gamma,
                    #         lam=self.config.algorithm.lam,
                    #         num_repeat=self.config.actor_rollout_ref.rollout.n,
                    #         norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    #         multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    #         use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    #         pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    #         pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                    #     )
                assert isinstance(batch, DataProto), "batch should be a DataProto object before backward pass"

                with _timer("update_actor", timing_raw):
                    print("Start updating actor")
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    print("Actor updated successfully")
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with _timer("dump_rollout_generations", timing_raw):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict={},
                            dump_path=rollout_data_dir,
                        )
                        
                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()


            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1
            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return