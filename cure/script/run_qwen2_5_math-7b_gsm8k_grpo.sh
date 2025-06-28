set -x

# This script is almost the same as the official sglang worker tutorial: https://verl.readthedocs.io/en/v0.4.x/workers/sglang_worker.html

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HOME/hfmodels/Qwen2.5-Math-7B \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.project_name='sglang_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2_5_math-7b_gsm8k_grpo' \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log

# TODO: launch_reward_fn_async