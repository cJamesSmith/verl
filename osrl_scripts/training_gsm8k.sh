#!/bin/bash
set -x

export WANDB_MODE=offline
# export VLLM_ATTENTION_BACKEND=XFORMERS  # vllm>=0.8
gsm8k_train_path=data/gsm8k/train.parquet
gsm8k_test_path=data/gsm8k/test.parquet
math_train_path=data/math/train.parquet
math_test_path=data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# This setup is for a 8x80G node
# be careful about actor_rollout_ref.rollout.val_kwargs.n
python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/gsm8k/train.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 data.filter_overlong_prompts=True \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='/home/aiops/chenxw/hfmodels/Qwen2.5-Math-7B' \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.checkpoint.contents="['model', 'optimizer', 'extra', 'hf_model']" \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.free_cache_engine=False \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.val_kwargs.n=8 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl_few_shot'\
 trainer.experiment_name=Qwen2.5-Math-7B-gsm\
 trainer.val_before_train=True \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.max_actor_ckpt_to_keep=1 \
 trainer.test_freq=1 \
 trainer.total_epochs=50