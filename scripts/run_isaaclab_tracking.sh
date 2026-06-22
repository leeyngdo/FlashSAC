#!/bin/bash
# IsaacLab G1 motion-tracking recipe.

# .npz path, list, or directory.
MOTION_FILES='[/home/ubuntu/youngdo/FlashSAC/flash_rl/envs/isaaclab_envs/motions/TODO_REPLACE_ME.npz]'

seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    uv run python train.py \
        --config_name flashSAC_base \
        --overrides seed=${seed} \
        --overrides env=isaaclab_tracking \
        --overrides env.env_name=Isaac-Tracking-Flat-G1-v0 \
        --overrides "env.motion.motion_files=${MOTION_FILES}" \
        --overrides num_env_steps=512_000_000 \
        --overrides num_train_envs=1024 \
        --overrides num_eval_envs=null \
        --overrides num_record_envs=null \
        --overrides num_eval_episodes=1024 \
        --overrides num_record_episodes=0 \
        --overrides agent=flashSAC \
        --overrides agent.buffer_max_length=10_000_000 \
        --overrides agent.buffer_min_length=100_000 \
        --overrides agent.buffer_device_type='cuda' \
        --overrides agent.sample_batch_size=2048 \
        --overrides agent.use_amp=true \
        --overrides updates_per_interaction_step=2 \
        --overrides agent.asymmetric_observation=true \
        --overrides gamma=0.99 \
        --overrides n_step=3
done
