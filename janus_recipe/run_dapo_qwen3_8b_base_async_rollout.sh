#!/usr/bin/env bash
set -xeuo pipefail

entity="stevez25-hong-kong-polytechnic-university"
project_name='ReLaX-text'
# MODEL_ID="Janus/HF-Qwen3-8B-tulumix-sft"
# exp_name='1-16-HF-Qwen3-8B-tulumix-sft'
MODEL_ID="Qwen3-4B-Base"
exp_name='aiic_debug-Qwen3-4B-Base-async-rollout'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 12))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=4
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=16
train_prompt_mini_bsz=$((train_prompt_bsz*n_resp_per_prompt))
# train_prompt_mini_bsz=512

NNODES=${NNODES:-1}

# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
rollout_name="vllm" # sglang or vllm
return_raw_chat="True"
if [ "$rollout_name" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

MODEL_HOME=${MODEL_HOME:-"hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models"}
DATASETS_HOME=${DATASETS_HOME:-"/mnt/hdfs/tiktok_aiic/user/chenxianwei/datasets"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_HOME}/${MODEL_ID}"}
CKPTS_DIR=${CKPTS_DIR:-"/mnt/hdfs/tiktok_aiic/user/chenxianwei/checkpoints/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${DATASETS_HOME}/relax/ReLaX_MATH/data/train-00000-of-00001.parquet"}
TEST_FILE="['${DATASETS_HOME}/relax/Mathematics_Eval/data/AIME24-00000-of-00001.parquet','${DATASETS_HOME}/relax/Mathematics_Eval/data/AIME25-00000-of-00001.parquet','${DATASETS_HOME}/relax/Mathematics_Eval/data/AMC22-00000-of-00001.parquet','${DATASETS_HOME}/relax/Mathematics_Eval/data/AMC23-00000-of-00001.parquet']"

# Algorithm
temperature=1
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=1
fsdp_size=-1

TRAIN_CMD=(
    python3 -m math_recipe.main_dapo
        data.train_files="${TRAIN_FILE}"
        data.val_files="${TEST_FILE}"
        data.prompt_key=prompt
        data.truncation='error'
        # data.filter_overlong_prompts=True
        # data.filter_overlong_prompts_workers=16
        data.return_raw_chat=$return_raw_chat
        data.max_prompt_length=${max_prompt_length}
        data.max_response_length=${max_response_length}
        data.gen_batch_size=${gen_prompt_bsz}
        data.train_batch_size=${train_prompt_bsz}
        actor_rollout_ref.rollout.n=${n_resp_per_prompt}
        algorithm.adv_estimator=${adv_estimator}
        algorithm.use_kl_in_reward=${use_kl_in_reward}
        algorithm.kl_ctrl.kl_coef=${kl_coef}
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
        actor_rollout_ref.actor.clip_ratio_c=10.0
        algorithm.filter_groups.enable=${enable_filter_groups}
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
        algorithm.filter_groups.metric=${filter_groups_metric}
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
        actor_rollout_ref.model.path="${MODEL_PATH}"
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.optim.lr_warmup_steps=10
        actor_rollout_ref.actor.optim.weight_decay=0.1
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload}
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}
        actor_rollout_ref.actor.entropy_coeff=0
        actor_rollout_ref.actor.grad_clip=1.0
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80
        config.actor_rollout_ref.rollout.enforce_eager = True
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
        actor_rollout_ref.rollout.temperature=${temperature}
        actor_rollout_ref.rollout.top_p=${top_p}
        actor_rollout_ref.rollout.top_k="${top_k}"
        actor_rollout_ref.rollout.val_kwargs.temperature=0
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
        actor_rollout_ref.rollout.val_kwargs.do_sample=False
        actor_rollout_ref.rollout.val_kwargs.n=1
        actor_rollout_ref.rollout.name=$rollout_name
        actor_rollout_ref.rollout.mode=$rollout_mode
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload}
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
        reward_model.use_reward_loop=True
        reward_model.reward_manager=dapo
        reward_model.num_workers=2
        custom_reward_function.path=tests/experimental/reward_loop/reward_fn.py
        custom_reward_function.name=compute_score_math_verify
        reward_model.overlong_buffer.enable=${enable_overlong_buffer}
        reward_model.overlong_buffer.len=${overlong_buffer_len}
        reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}
        trainer.logger='["console"]'
        +trainer.entity="${entity}"
        trainer.project_name="${project_name}"
        trainer.experiment_name="${exp_name}"
        trainer.n_gpus_per_node=2
        trainer.nnodes="${NNODES}"
        trainer.val_before_train=False
        trainer.test_freq=5
        trainer.save_freq=-1
        trainer.total_epochs=3
        trainer.default_local_dir="${CKPTS_DIR}"
        trainer.resume_mode=auto
        trainer.total_training_steps=500
        # trainer.validation_data_dir="/mnt/hdfs/tiktok_aiic/user/chenxianwei/validation_data/${project_name}/${exp_name}"
        trainer.val_only=False
        trainer.max_actor_ckpt_to_keep=1
        trainer.log_val_generations=1
        trainer.rollout_data_dir="/mnt/hdfs/tiktok_aiic/user/chenxianwei/rollout_data/${project_name}/${exp_name}" \
        "$@"
)


"${TRAIN_CMD[@]}"
