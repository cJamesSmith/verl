set -x

export VLLM_USE_V1=1

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --entrypoint) entrypoint="$2"; shift 2 ;;
        --config_name) config_name="$2"; shift 2 ;;
        --project_name) project_name="$2"; shift 2 ;;
        --exp_name) exp_name="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --data_path) data_path="$2"; shift 2 ;;
        --train_data)
            shift
            while [[ "$#" -gt 0 && ! $1 =~ ^-- ]]; do
                train_data="${train_data} $1"
                shift
            done
            ;;
        --test_data)
            shift
            while [[ "$#" -gt 0 && ! $1 =~ ^-- ]]; do
                test_data="${test_data} $1"
                shift
            done
            ;;
        --data_custom_cls_path) data_custom_cls_path="$2"; shift 2 ;;
        --data_custom_cls_name) data_custom_cls_name="$2"; shift 2 ;;
        --reward_manager) reward_manager="$2"; shift 2 ;;
        --custom_reward_path) custom_reward_path="$2"; shift 2 ;;
        --custom_reward_name) custom_reward_name="$2"; shift 2 ;;
        --tool_config_path) tool_config_path="$2"; shift 2 ;;
        --interaction_config_path) interaction_config_path="$2"; shift 2 ;;
        --agent_loop_config_path) agent_loop_config_path="$2"; shift 2 ;;
        --ckpt_path) ckpt_path="$2"; shift 2 ;;
        --adv_estimator) adv_estimator="$2"; shift 2 ;;
        --max_turns) max_turns="$2"; shift 2 ;;
        --max_prompt_length) max_prompt_length="$2"; shift 2 ;;
        --max_response_length) max_response_length="$2"; shift 2 ;;
        --train_batch_size) train_batch_size="$2"; shift 2 ;;
        --ppo_mini_batch_size) ppo_mini_batch_size="$2"; shift 2 ;;
        --n_resp_per_prompt) n_resp_per_prompt="$2"; shift 2 ;;
        --n_resp_per_prompt_val) n_resp_per_prompt_val="$2"; shift 2 ;;
        --val_before_train) val_before_train="$2"; shift 2 ;;
        --val_max_samples) val_max_samples="$2"; shift 2 ;;
        --val_only) val_only="$2"; shift 2 ;;
        --train_sp) train_sp="$2"; shift 2 ;;
        --train_tp) train_tp="$2"; shift 2 ;;
        --train_ep) train_ep="$2"; shift 2 ;;
        --train_pp) train_pp="$2"; shift 2 ;;
        --infer_tp) infer_tp="$2"; shift 2 ;;
        --gen_ep) gen_ep="$2"; shift 2 ;;
        --actor_max_token_len_per_gpu) actor_max_token_len_per_gpu="$2"; shift 2 ;;
        --log_prob_max_token_len_per_gpu) log_prob_max_token_len_per_gpu="$2"; shift 2 ;;
        --total_epochs) total_epochs="$2"; shift 2 ;;
        --save_freq) save_freq="$2"; shift 2 ;;
        --test_freq) test_freq="$2"; shift 2 ;;
        --gpu_memory_utilization) gpu_memory_utilization="$2"; shift 2 ;;
        --actor_lr) actor_lr="$2"; shift 2 ;;
        --offload) offload="$2"; shift 2 ;;
        --use_kl_loss) use_kl_loss="$2"; shift 2 ;;
        --kl_loss_coef) kl_loss_coef="$2"; shift 2 ;;
        --use_kl_in_reward) use_kl_in_reward="$2"; shift 2 ;;
        --kl_coef) kl_coef="$2"; shift 2 ;;
        --clip_ratio_low) clip_ratio_low="$2"; shift 2 ;;
        --clip_ratio_high) clip_ratio_high="$2"; shift 2 ;;
        --filter_groups_enable) filter_groups_enable="$2"; shift 2 ;;
        --filter_groups_metric) filter_groups_metric="$2"; shift 2 ;;
        --filter_groups_max_num_gen_batches) filter_groups_max_num_gen_batches="$2"; shift 2 ;;
        # PPO-specific critic arguments
        --critic_model_path) critic_model_path="$2"; shift 2 ;;
        --critic_lr) critic_lr="$2"; shift 2 ;;
        --critic_sp) critic_sp="$2"; shift 2 ;;
        --critic_offload) critic_offload="$2"; shift 2 ;;
        --ppo_micro_batch_size_per_gpu) ppo_micro_batch_size_per_gpu="$2"; shift 2 ;;
        --critic_max_token_len_per_gpu) critic_max_token_len_per_gpu="$2"; shift 2 ;;
        --critic_warmup) critic_warmup="$2"; shift 2 ;;
        *) break ;;
    esac
done

# ================= data/model/reward =================
# Helper function to format space-separated files into a Python list ['path1/file1', 'path2/file2']
function format_files() {
    local base_path=$1
    local file_list=$2
    local result="["
    local first=true
    for f in $file_list; do
        if [ "$first" = true ]; then
            result+="'$base_path/$f'"
            first=false
        else
            result+=", '$base_path/$f'"
        fi
    done
    result+="]"
    echo "$result"
}

train_files=$(format_files "$data_path" "$train_data")
test_files=$(format_files "$data_path" "$test_data")

TRAIN_CMD=(
    python3 -m ${entrypoint:-"verl.trainer.main_ppo"}
    --config-path="${config_path:-"config"}"
    --config-name=${config_name:-"ppo_trainer.yaml"}
    
    # algorithm
    algorithm.adv_estimator=${adv_estimator:-grpo}
    algorithm.use_kl_in_reward=${use_kl_in_reward:-False}
    algorithm.kl_ctrl.kl_coef=${kl_coef:-0.0}
    
    # data
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.val_max_samples=${val_max_samples:--1}
    data.return_raw_chat=True
    data.train_batch_size=${train_batch_size:-64}
    data.max_prompt_length=${max_prompt_length:-4096}
    data.max_response_length=${max_response_length:-16384}
    data.filter_overlong_prompts=True
    data.truncation='error'
    
    # reward model
    reward_model.reward_manager=${reward_manager:-"naive"}
    
    # actor_rollout_ref
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss:-False}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef:-0.0}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low:-0.2}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high:-0.28}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.optim.lr=${actor_lr:-1e-6}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size:-64}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu:-16384}
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp:-4}
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload:-True}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload:-True}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu:-16384}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.strategy=fsdp
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp:-4}
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload:-True}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp:-4}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization:-0.7}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val:-4}
    
    # trainer
    trainer.logger=['console','wandb']
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=$NGPUS_PER_NODE
    trainer.val_before_train=${val_before_train:-False}
    trainer.log_val_generations=0
    trainer.nnodes=$NNODES
    trainer.save_freq=${save_freq:-20}
    trainer.default_local_dir=$ckpt_path/$exp_name
    trainer.test_freq=${test_freq:-10}
    trainer.total_epochs=${total_epochs:-50}
    trainer.val_only=${val_only:-False}
    "$@"
)

if [ -n "$filter_groups_enable" ]; then
    TRAIN_CMD+=(
        +algorithm.filter_groups.enable=${filter_groups_enable}
        +algorithm.filter_groups.metric=${filter_groups_metric:-acc}
        +algorithm.filter_groups.max_num_gen_batches=${filter_groups_max_num_gen_batches:-0}
    )
fi

if [ -n "$data_custom_cls_path" ]; then
    TRAIN_CMD+=(
        data.custom_cls.path=${data_custom_cls_path}
        data.custom_cls.name=${data_custom_cls_name}
    )
fi

if [ -n "$custom_reward_path" ]; then
    TRAIN_CMD+=(
        custom_reward_function.path=${custom_reward_path}
        custom_reward_function.name=${custom_reward_name}
    )
fi

if [ "${max_turns:-0}" -gt 1 ]; then
    TRAIN_CMD+=(
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns:-16}
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns:-16}
        actor_rollout_ref.rollout.multi_turn.format=hermes
    )
    if [ -n "$tool_config_path" ]; then
        TRAIN_CMD+=(actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path})
    fi
    if [ -n "$interaction_config_path" ]; then
        TRAIN_CMD+=(actor_rollout_ref.rollout.multi_turn.interaction_config_path=$interaction_config_path)
    fi
    if [ -n "$agent_loop_config_path" ]; then
        TRAIN_CMD+=(actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path)
    fi
fi

# Add critic configuration for PPO (GAE advantage estimator)
if [[ "${adv_estimator:-grpo}" == "gae" || "${adv_estimator:-grpo}" == "turn_gae" ]]; then
    TRAIN_CMD+=(
        # PPO typically uses n=1 (single response per prompt) since it uses a learned value function
        actor_rollout_ref.rollout.n=${n_resp_per_prompt:-1}
        critic.optim.lr=${critic_lr:-1e-5}
        critic.model.use_remove_padding=True
        critic.model.path=${critic_model_path:-$model_path}
        critic.model.enable_gradient_checkpointing=True
        critic.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu:-1}
        critic.ulysses_sequence_parallel_size=${critic_sp:-${train_sp:-4}}
        critic.model.fsdp_config.param_offload=${critic_offload:-${offload:-True}}
        critic.model.fsdp_config.optimizer_offload=${critic_offload:-${offload:-True}}
        critic.use_dynamic_bsz=True
        critic.ppo_max_token_len_per_gpu=${critic_max_token_len_per_gpu:-${actor_max_token_len_per_gpu:-16384}}
        trainer.critic_warmup=${critic_warmup:-0}
    )
else
    # GRPO needs multiple responses per prompt for group-based ranking
    TRAIN_CMD+=(
        actor_rollout_ref.rollout.n=${n_resp_per_prompt:-16}
    )
fi

"${TRAIN_CMD[@]}"