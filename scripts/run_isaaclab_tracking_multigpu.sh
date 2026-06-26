#!/bin/bash
##################################################################################
# IsaacLab Motion Tracking - Multi-GPU (data-parallel) training
#
# Launches one process per GPU with torchrun. Each rank runs the full num_train_envs
# on its own GPU with its own replay buffer; gradients are averaged across ranks.
# Therefore, with N GPUs the effective envs/batch scale by N:
#   - effective envs       = num_train_envs * N
#   - effective batch size = sample_batch_size * N
# `num_env_steps` remains the global env-step budget; train.py divides interaction
# steps across ranks. Learning rate is NOT scaled.
##################################################################################

# Number of GPUs to use (defaults to all visible GPUs).
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}

# Hydra list literal syntax: ["/path/a.npz","/path/b.npz"] or a directory "/path/motions".
: "${MOTION_FILES:?Set MOTION_FILES to a Hydra list literal (e.g. '[\"/path/a.npz\"]') or a directory of .npz clips}"

seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" train.py \
        --config_name flashSAC_base \
        --overrides seed=${seed} \
        --overrides env=isaaclab_tracking \
        --overrides env.env_name=Isaac-Tracking-Flat-G1-v0 \
        --overrides "env.motion.motion_files=${MOTION_FILES}" \
        --overrides num_env_steps=204_800_000 \
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
