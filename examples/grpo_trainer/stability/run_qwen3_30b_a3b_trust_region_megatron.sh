#!/usr/bin/env bash
# Trust-region/off-policy method sweep | Qwen3-30B-A3B-Base | Megatron training | NVIDIA GPUs
#
# METHOD selects the algorithm configuration:
#   grpo | cispo | trm | minpro | dppo | dppo_tv | dppo_kl | cppo | sis
#
# Data paths are intentionally supplied by environment variables so a deduped
# DAPO-Math-17k parquet can be reused without hard-coding local paths:
#   TRAIN_FILE=/path/to/dapo_math_17k_dedup.parquet \
#   VAL_FILE=/path/to/aime_or_dapo_val.parquet \
#   METHOD=trm bash examples/grpo_trainer/run_qwen3_30b_a3b_trust_region_megatron.sh

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}

########################### user-adjustable ###########################
METHOD=${METHOD:-grpo}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}

TRAIN_FILE=${TRAIN_FILE:?Please set TRAIN_FILE to the deduplicated DAPO-Math-17k parquet path}
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-30720}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}
actor_ep=${ACTOR_EP:-8}
actor_etp=${ACTOR_ETP:-1}
actor_cp=${ACTOR_CP:-1}
all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
rollout_n=${ROLLOUT_N:-16}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-10240}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-10240}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}

ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

# Shared GRPO-family defaults. Individual METHOD cases override these.
adv_estimator=grpo
policy_loss_mode=vanilla
loss_agg_mode=token-mean
use_kl_loss=False
kl_loss_coef=0.0
norm_adv_by_std_in_grpo=False
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}

trm_max_kl=${TRM_MAX_KL:-0.05}
trm_avg_kl=${TRM_AVG_KL:-null}
cppo_token_kl=${CPPO_TOKEN_KL:-0.05}
cppo_prefix_kl=${CPPO_PREFIX_KL:-0.01}
cppo_position_weight_power=${CPPO_POSITION_WEIGHT_POWER:-1.0}
sis_envelope=${SIS_ENVELOPE:-2.0}
sis_acceptance=${SIS_ACCEPTANCE:-deterministic}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-10}
val_before_train=${VAL_BEFORE_TRAIN:-False}
log_val_generations=${LOG_VAL_GENERATIONS:-0}

project_name=${PROJECT_NAME:-verl_trust_region_qwen3_moe}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_${METHOD}_vllm_megatron}
########################### end user-adjustable ###########################

case "${METHOD}" in
    grpo)
        policy_loss_mode=vanilla
        use_kl_loss=${USE_KL_LOSS:-False}
        kl_loss_coef=${KL_LOSS_COEF:-0.0}
        ;;
    cispo)
        policy_loss_mode=cispo
        clip_ratio_low=${CLIP_RATIO_LOW:-10.0}
        clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}
        use_kl_loss=${USE_KL_LOSS:-True}
        kl_loss_coef=${KL_LOSS_COEF:-0.001}
        ;;
    trm)
        policy_loss_mode=trm
        ;;
    minpro)
        policy_loss_mode=minpro
        clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
        clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
        ;;
    dppo | dppo_tv)
        policy_loss_mode=dppo_tv
        clip_ratio_low=${CLIP_RATIO_LOW:-0.15}
        clip_ratio_high=${CLIP_RATIO_HIGH:-0.15}
        clip_ratio_c=${CLIP_RATIO_C:-10000.0}
        loss_agg_mode=seq-mean-token-sum-norm
        ;;
    dppo_kl)
        policy_loss_mode=dppo_kl
        clip_ratio_low=${CLIP_RATIO_LOW:-0.05}
        clip_ratio_high=${CLIP_RATIO_HIGH:-0.05}
        clip_ratio_c=${CLIP_RATIO_C:-10000.0}
        loss_agg_mode=seq-mean-token-sum-norm
        ;;
    cppo)
        policy_loss_mode=cppo
        ;;
    sis)
        policy_loss_mode=sis
        use_kl_loss=${USE_KL_LOSS:-True}
        kl_loss_coef=${KL_LOSS_COEF:-0.001}
        ;;
    *)
        echo "Unknown METHOD=${METHOD}. Expected: grpo | cispo | trm | minpro | dppo | dppo_tv | dppo_kl | cppo | sis" >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo}
)

REWARD=(
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
)

DATA=(
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${train_batch_size}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode}
    actor_rollout_ref.actor.policy_loss.trm_max_kl=${trm_max_kl}
    actor_rollout_ref.actor.policy_loss.trm_avg_kl=${trm_avg_kl}
    actor_rollout_ref.actor.policy_loss.cppo_token_kl=${cppo_token_kl}
    actor_rollout_ref.actor.policy_loss.cppo_prefix_kl=${cppo_prefix_kl}
    actor_rollout_ref.actor.policy_loss.cppo_position_weight_power=${cppo_position_weight_power}
    actor_rollout_ref.actor.policy_loss.sis_envelope=${sis_envelope}
    actor_rollout_ref.actor.policy_loss.sis_acceptance=${sis_acceptance}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
    actor_rollout_ref.rollout.response_length=${max_response_length}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ref_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${val_before_train}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.log_val_generations=${log_val_generations}
)

EXTRA=(
    model_engine=megatron
)

if [ -n "${MCORE_MODEL_PATH}" ]; then
    EXTRA+=(
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
