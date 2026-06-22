#!/bin/bash
##################################################################################
# IsaacLab Motion Tracking (G1, GPU Simulator)
#
# Reward-group sweep for the vendored BeyondMimic G1 tracking task.
# Mirrors scripts/run_isaaclab.sh GPU recipe (num_train_envs=1024, cuda buffer,
# batch 2048, amp) plus the tracking-specific knobs (asymmetric obs, grouped
# reward overrides). Edit MOTION_FILES below before running.
##################################################################################

# REQUIRED: list of .npz motion clips OR a directory of .npz (or a single path).
# Hydra list literal syntax: ["/path/a.npz","/path/b.npz"] or a directory "/path/motions".
# TODO: replace the placeholder with your motion clip path(s).
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
