# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
DATA_DIR=$1
CONFIG_PATH="$PROJECT_DIR/examples/video_tools/config"
CKPT_DIR=$2
export RAY_DEBUG_POST_MORTEM=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=10

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='timer1_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.sglang_video=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=$CKPT_DIR/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='geo3k_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-geo3k-sgl-multi-w-tool-verify-n16' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=$DATA_DIR/TimeR1/timer1_train.parquet \
    data.val_files=$DATA_DIR/TimeR1/timer1_train.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/video_tools/config/mcp_tool_config.yaml" \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=outputs/rollout_data \
    custom_reward_function.path=custom_rewards/tool_reward.py \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable

