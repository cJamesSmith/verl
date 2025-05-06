#!/bin/bash
set -x

CHECKPOINTS_DIR=ckpt # TODO: change to your own path

# export VLLM_ATTENTION_BACKEND=XFORMERS  # vllm>=0.8

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/one_shot_rlvr/pi1_r128.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-7B' \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.checkpoint.contents=['model'] \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.free_cache_engine=False \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console'] \
 trainer.project_name='verl_few_shot'\
 trainer.experiment_name='Qwen2.5-Math-7B-pi1_r128-verl'\
 trainer.val_before_train=True \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=50 \
 trainer.test_freq=5 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=1000 2>&1 | tee verl_demo.log