# AIIC RL Recipe

[**Lark Documentation**](https://bytedance.larkoffice.com/wiki/Q7RVwcH0eiIVPOkwChbcthcWnyp)

## Installation

```bash
sudo python3 -m pip uninstall bytedray -y && sudo python3 -m pip install --force-reinstall "ray[data,train,tune,serve]"
sudo python3 -m pip uninstall grpcio -y && sudo python3 -m pip install grpcio==1.62.1
sudo python3 -m pip uninstall byted-wandb -y && sudo python3 -m pip install wandb==0.23.1
sudo python3 -m pip uninstall verl -y
sudo python3 -m pip install protobuf==4.25.3
sudo python3 -m pip install numba==0.63.1
sudo python3 -m pip install sandbox_fusion
sudo python3 -m pip install logfire
sudo python3 -m pip install pydantic-core==2.41.5

# Install firejail and sandbox dependencies
sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install firejail
sudo python3 -m pip install "fastapi[all]" uvicorn

export WANDB_API_KEY=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
# export LOGFIRE_KEY=pylf_v1_us_WDr56TvhGPcGq7Sllx7mvHklQy4Wg2rQPYC93cYWRGFb
export NGPUS_PER_NODE=8
export NNODES=$ARNOLD_WORKER_NUM
export NODE_RANK=$ARNOLD_ID
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=$(echo $ARNOLD_WORKER_0_PORT | cut -d',' -f1)

# For fully async only
export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
```

### SWE-Bench Setup

```bash
git clone https://github.com/OpenHands/OpenHands.git
cd OpenHands
sudo python3 -m pip install -e .
cd ..

git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
sudo python3 -m pip install -e .
cd ..
```

### Local Python Sandbox Setup (Firejail)

```bash
# To start python sandbox:
bash aiic_recipe/start_sandbox.sh

# To start terminal sandbox:
bash aiic_recipe/start_sandbox.sh --type terminal
```

## Training-Inference Colocated Training

### Single-Turn

#### Code RL

```bash
# Dense reward (passed/total)
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name code_rl \
    --exp_name debug-code-single-turn-qwen3-4b-instruct-2507-16k \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name SingleTurnCodeRLDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/code_rl_checkpoints \
    --max_prompt_length 4096 \
    --max_response_length 12288 \
    --train_sp 2 \
    --actor_max_token_len_per_gpu 8192 \
    --log_prob_max_token_len_per_gpu 16384 \
    --filter_groups_enable True \
    --filter_groups_metric seq_reward \
    --filter_groups_max_num_gen_batches 0 \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="false" \
    "algorithm.rollout_correction.rollout_rs='token_k1,seq_max_k2'" \
    "algorithm.rollout_correction.rollout_rs_threshold='0.6_1.6,2.5'" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="ppo_clip" \
    +reward_model.max_concurrent=200

# Binary reward (0/1)
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name code_rl \
    --exp_name debug-code-single-turn-qwen3-4b-instruct-2507-binary-16k \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name SingleTurnCodeRLDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_binary_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/code_rl_checkpoints \
    --max_prompt_length 4096 \
    --max_response_length 12288 \
    --train_sp 2 \
    --actor_max_token_len_per_gpu 8192 \
    --log_prob_max_token_len_per_gpu 16384 \
    --filter_groups_enable True \
    --filter_groups_metric seq_reward \
    --filter_groups_max_num_gen_batches 0 \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="false" \
    "algorithm.rollout_correction.rollout_rs='token_k1,seq_max_k2'" \
    "algorithm.rollout_correction.rollout_rs_threshold='0.6_1.6,2.5'" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="ppo_clip" \
    +reward_model.max_concurrent=200

# PPO with GAE (requires critic model)
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name code_rl \
    --exp_name debug-code-single-turn-qwen3-4b-instruct-2507-ppo \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name SingleTurnCodeRLDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/code_rl_checkpoints \
    --max_prompt_length 4096 \
    --max_response_length 12288 \
    --train_sp 2 \
    --actor_max_token_len_per_gpu 8192 \
    --log_prob_max_token_len_per_gpu 16384 \
    --adv_estimator gae \
    --critic_lr 1e-5 \
    --critic_sp 2 \
    --train_batch_size 1024 \
    --ppo_mini_batch_size 1024 \
    +reward_model.max_concurrent=200
```

### Multi-Turn

#### Math RL with Code Interpreter (ReTool)

```bash
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name retool \
    --exp_name dapo-qwen3-4b-instruct-2507 \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data DAPO-Math-17k \
    --test_data AIME_2024 aime_2025 \
    --data_custom_cls_path aiic_recipe/dataset/retool.py \
    --data_custom_cls_name ReToolDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/retool.py \
    --custom_reward_name compute_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/retool_checkpoints \
    --max_prompt_length 2048 \
    --max_response_length 16384 \
    --max_turns 16 \
    --tool_config_path aiic_recipe/config/sandbox_fusion_tool_config.yaml \
    --total_epochs 1 \
    --actor_max_token_len_per_gpu 18432 \
    --log_prob_max_token_len_per_gpu 36864 \
    --filter_groups_enable True \
    --filter_groups_metric seq_reward \
    --filter_groups_max_num_gen_batches 0 \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="false" \
    "algorithm.rollout_correction.rollout_rs='token_k1,seq_max_k2'" \
    "algorithm.rollout_correction.rollout_rs_threshold='0.6_1.6,2.5'" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="ppo_clip" \
    +reward_model.max_concurrent=200
```

#### Code RL with Code Interpreter

```bash
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name code_rl \
    --exp_name debug-code-qwen3-4b-instruct-2507-2turns-grpo \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name MultiTurnCodeRLDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/code_rl_checkpoints \
    --interaction_config_path aiic_recipe/config/multi_turn_interaction_config.yaml \
    --agent_loop_config_path aiic_recipe/config/multi_turn_agent_loop_config.yaml \
    --max_turns 2 \
    --max_prompt_length 4096 \
    --max_response_length 16384 \
    --train_sp 2 \
    --actor_max_token_len_per_gpu 10240 \
    --log_prob_max_token_len_per_gpu 20480 \
    --filter_groups_enable True \
    --filter_groups_metric seq_reward \
    --filter_groups_max_num_gen_batches 0 \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="false" \
    "algorithm.rollout_correction.rollout_rs='token_k1,seq_max_k2'" \
    "algorithm.rollout_correction.rollout_rs_threshold='0.6_1.6,2.5'" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="ppo_clip" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=aiic_recipe.agent_loop.agent_loop.AgentLoopManagerWithTurnID \
    +reward_model.max_concurrent=200

# Turn RLOO with delta reward shaping
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name code_rl \
    --exp_name debug-code-qwen3-4b-instruct-2507-2turns-turnrloo \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name MultiTurnCodeRLRewardShapingDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/code_rl_checkpoints \
    --interaction_config_path aiic_recipe/config/multi_turn_interaction_config.yaml \
    --agent_loop_config_path aiic_recipe/config/multi_turn_agent_loop_config.yaml \
    --adv_estimator turn_rloo \
    --max_turns 2 \
    --max_prompt_length 4096 \
    --max_response_length 16384 \
    --train_sp 2 \
    --actor_max_token_len_per_gpu 10240 \
    --log_prob_max_token_len_per_gpu 20480 \
    --filter_groups_enable True \
    --filter_groups_metric seq_reward \
    --filter_groups_max_num_gen_batches 0 \
    algorithm.rollout_correction.rollout_is="token" \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize="false" \
    "algorithm.rollout_correction.rollout_rs='token_k1,seq_max_k2'" \
    "algorithm.rollout_correction.rollout_rs_threshold='0.6_1.6,2.5'" \
    algorithm.rollout_correction.bypass_mode="true" \
    algorithm.rollout_correction.loss_type="ppo_clip" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=aiic_recipe.agent_loop.agent_loop.AgentLoopManagerWithTurnID \
    +reward_model.max_concurrent=200

# Turn GAE with critic (turn-level value function)
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train.sh \
    --entrypoint aiic_recipe.main \
    --config_name trainer.yaml \
    --project_name code_rl \
    --exp_name debug-code-qwen3-4b-instruct-2507-2turns-turngae \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name MultiTurnCodeRLRewardShapingDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_score \
    --ckpt_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/code_rl_checkpoints \
    --interaction_config_path aiic_recipe/config/multi_turn_interaction_config.yaml \
    --agent_loop_config_path aiic_recipe/config/multi_turn_agent_loop_config.yaml \
    --adv_estimator turn_gae \
    --critic_lr 1e-5 \
    --critic_sp 2 \
    --max_turns 2 \
    --max_prompt_length 4096 \
    --max_response_length 16384 \
    --train_sp 2 \
    --actor_max_token_len_per_gpu 10240 \
    --log_prob_max_token_len_per_gpu 20480 \
    --train_batch_size 1024 \
    --ppo_mini_batch_size 1024 \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=aiic_recipe.agent_loop.agent_loop.AgentLoopManagerWithTurnID \
    +reward_model.max_concurrent=200
```

## Fully Async Training

```bash
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train_fully_async.sh \
    --project_name DAPO \
    --exp_name debug-qwen3-4b-instruct-2507-math-fully-async \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data DAPO-Math-17k/data/dapo-math-17k.parquet \
    --test_data BytedTsinghua-SIA/AIME_2024/data/aime-2024.parquet \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --max_prompt_length 2048 \
    --max_response_length 8192 \
    --reward_manager dapo \
    --actor_ppo_max_token_len 10240 \
    --infer_ppo_max_token_len 16384 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=8192

    # +algorithm.filter_groups.enable=True \
    # +algorithm.filter_groups.metric=acc \
    # +algorithm.filter_groups.max_num_gen_batches=0 \
```

To enable rollout correction, add the following arguments:

```bash

    +async_training.compute_prox_log_prob=True \
    +algorithm.rollout_correction.rollout_is=token \
    +algorithm.rollout_correction.rollout_is_threshold=2.0 \
    +algorithm.rollout_correction.rollout_rs=seq_mean_k1 \
    +algorithm.rollout_correction.rollout_rs_threshold=0.99_1.001
```

```bash
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train_fully_async.sh \
    --project_name ppo \
    --exp_name debug-qwen3-4b-instruct-2507-code-fully-async \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data leetcode/train.jsonl \
    --test_data leetcode/test.jsonl \
    --data_custom_cls_path aiic_recipe/dataset/code.py \
    --data_custom_cls_name SingleTurnCodeRLDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/code.py \
    --custom_reward_name compute_score \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --max_prompt_length 4096 \
    --max_response_length 32768 \
    --sp_size 4 \
    --actor_ppo_max_token_len 10240 \
    --infer_ppo_max_token_len 10240 \
    --staleness_threshold 0.5 \
    +reward_model.max_concurrent=200

    # +algorithm.filter_groups.enable=True \
    # +algorithm.filter_groups.metric=acc \
    # +algorithm.filter_groups.max_num_gen_batches=0 \
```

```bash
bash aiic_recipe/start_sandbox.sh

bash aiic_recipe/start_ray.sh aiic_recipe/train_fully_async.sh \
    --project_name retool \
    --exp_name debug-qwen3-4b-instruct-2507-retool-fully-async \
    --data_path /mnt/hdfs/tiktok_aiic/user/longtao.zheng/rl_datasets \
    --train_data DAPO-Math-17k \
    --test_data AIME_2024 aime_2025 \
    --data_custom_cls_path aiic_recipe/dataset/retool.py \
    --data_custom_cls_name ReToolDataset \
    --reward_manager rate_limited \
    --custom_reward_path aiic_recipe/reward_score/retool.py \
    --custom_reward_name compute_score \
    --model_path hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-4B-Instruct-2507 \
    --max_prompt_length 2048 \
    --max_response_length 16384 \
    --tool_config_path aiic_recipe/config/sandbox_fusion_tool_config.yaml \
    --max_turns 16 \
    +reward_model.max_concurrent=200

    # +algorithm.filter_groups.enable=True \
    # +algorithm.filter_groups.metric=acc \
    # +algorithm.filter_groups.max_num_gen_batches=0 \
```
