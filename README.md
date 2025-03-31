# GRPO Variants

Finetuning Qwen2.5-Math-1.5B.

### Baseline

`./examples/grpo_trainer/run_qwen2_5-1.5b_math.sh`:
```bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

gsm8k_train_path=$HOME/data/verl_origin/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/verl_origin/gsm8k/test.parquet
math_train_path=$HOME/data/verl_origin/math/train.parquet
math_test_path=$HOME/data/verl_origin/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HOME/data/models/Qwen2.5-Math-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo' \
    trainer.experiment_name='qwen2_1.5b_math_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
```

### Entropy Reward = 0

Intuition: Diversity of LLM's completions.

Modify `./examples/grpo_trainer/run_qwen2_5-1.5b_math.sh`:
```bash
...
    actor_rollout_ref.actor.entropy_coeff=0 \
...
```

### Group Reward Normalization Without std

Intuition: Unbiased estimation of RL objective function.

Modify `./verl/trainer/ppo/core_algos.py`:
```python
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):

...
...

        for i in range(bsz):
            # scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
```

### Token-Level Normalization

Intuition: Analyze loss function (advantage function) when using micro batch (assuming micro_bs = 2).

$$
\underset{\text{verl's grpo}}{\frac{1}{2}\left(\frac{A_1}{|o_1|}+\frac{A_2}{|o_2|}\right)} > \underset{\text{full batch size}}{\frac{A_1+A_2}{|o_1|+|o_2|}}
$$