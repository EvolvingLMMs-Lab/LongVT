# run on 8xH100
# make sure your current working directory is the root of the project
sleep 10
set -x

ulimit -n 65535

PROJECT_DIR="/pfs/training-data/sudongwang/VideoTool"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
VISUAL_DATASET_TRAIN_0_1_2=/pfs/training-data/sudongwang/dataset/deepeyes_process/data_0.1.2_visual_toolbox_v2_processed_100.parquet
VISUAL_DATASET_TRAIN_0_8=/pfs/training-data/sudongwang/dataset/deepeyes_process/data_v0.8_visual_toolbox_v2_processed_100.parquet
EUREKA_DATASET_TRAIN=/pfs/training-data/sudongwang/dataset/deepeyes_process/data_thinklite_reasoning_acc_processed_100.parquet

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geo3k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/pfs/training-data/sudongwang/model/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='zoom_in_tool_verl' \
    trainer.experiment_name='qwen2.5vl-3b-instruct-deepeyes' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=10000 \
    trainer.rollout_data_dir=/pfs/training-data/sudongwang/VideoTool/checkpoints/zoom_in_tool_verl/qwen2.5vl-3b-instruct-deepeyes/rollout \
    data.train_files=[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8}] \
    data.val_files=[${EUREKA_DATASET_TRAIN}]  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/pfs/training-data/sudongwang/VideoTool/examples/sglang_multiturn/config/tool_config/visual_toolbox_config.yaml \
    trainer.total_epochs=150 $@ | tee /pfs/training-data/sudongwang/VideoTool/checkpoints/zoom_in_tool_verl/qwen2.5vl-3b-instruct-deepeyes/tool_debug.log 